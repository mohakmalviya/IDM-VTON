import sys
sys.path.append('./')
sys.path.append('/')
from PIL import Image
import gradio as gr
import argparse, torch, os
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List
import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
import gc
import sys
import datetime
import platform
import subprocess

# Import bitsandbytes early for all quantization options
try:
    import bitsandbytes as bnb
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module
    bnb_available = True
except ImportError:
    bnb_available = False

# Global variable to store the original uploaded image (full resolution)
ORIGINAL_IMAGE = None

# Add command line arguments for VRAM optimization
parser = argparse.ArgumentParser()
parser.add_argument("--share", type=str, default=False, help="Set to True to share the app publicly.")
parser.add_argument("--lowvram", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--load_mode", default=None, type=str, choices=["4bit", "8bit"], help="Quantization mode for optimization memory consumption")
parser.add_argument("--fixed_vae", action="store_true", default=True, help="Use fixed vae for FP16.")
args = parser.parse_args()

load_mode = args.load_mode
fixed_vae = args.fixed_vae

dtype = torch.float16
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_id = 'yisol/IDM-VTON'
vae_model_id = 'madebyollin/sdxl-vae-fp16-fix'

dtypeQuantize = dtype

if load_mode == '8bit':
    dtypeQuantize = torch.float16  # Use fp16 instead of fp8 for 8-bit mode
elif load_mode == '4bit':
    if not bnb_available:
        raise ImportError("bitsandbytes is required for 4-bit quantization. Please install it with: pip install bitsandbytes")
    dtypeQuantize = torch.float16  # Use fp16 for computation with 4-bit weights

ENABLE_CPU_OFFLOAD = args.lowvram
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False
need_restart_cpu_offloading = False

# Initialize models as None for lazy loading
unet = None
pipe = None
UNet_Encoder = None
example_path = os.path.join(os.path.dirname(__file__), 'example')

# Utility functions for VRAM optimization
def torch_gc():
    """
    Garbage collection for torch CUDA memory
    """
    if torch.cuda.is_available():
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    gc.collect()

def restart_cpu_offload(pipe, load_mode):
    """
    Restart CPU offloading for pipeline
    """
    is_model_cpu_offload, is_sequential_cpu_offload = optionally_disable_offloading(pipe)
    gc.collect()
    torch.cuda.empty_cache()
    pipe.enable_model_cpu_offload()
    
    # Make sure encoder_hid_proj is on the same device as the unet
    if hasattr(pipe.unet, 'encoder_hid_proj') and pipe.unet.encoder_hid_proj is not None:
        pipe.unet.encoder_hid_proj.to(pipe.unet.device)

def optionally_disable_offloading(_pipeline):
    """
    Optionally removes offloading in case the pipeline has been already sequentially offloaded to CPU.
    """
    is_model_cpu_offload = False
    is_sequential_cpu_offload = False
    print("Restarting CPU Offloading for pipeline...")
    if _pipeline is not None:
        for _, component in _pipeline.components.items():
            if isinstance(component, torch.nn.Module) and hasattr(component, "_hf_hook"):
                if not is_model_cpu_offload:
                    is_model_cpu_offload = isinstance(component._hf_hook, CpuOffload)
                if not is_sequential_cpu_offload:
                    is_sequential_cpu_offload = isinstance(component._hf_hook, AlignDevicesHook)
               
                remove_hook_from_module(component, recurse=True)

    return (is_model_cpu_offload, is_sequential_cpu_offload)

def quantize_4bit(module):
    """
    Apply 4-bit quantization to model modules
    """
    if not bnb_available:
        raise ImportError("bitsandbytes is required for 4-bit quantization")
        
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            device = child.weight.data.device

            has_bias = child.bias is not None
            
            # fp16 for compute dtype leads to faster inference
            # and one should almost always use nf4 as a rule of thumb
            bnb_4bit_compute_dtype = torch.float16
            quant_type = "nf4"

            new_layer = bnb.nn.Linear4bit(
                in_features,
                out_features,
                bias=has_bias,
                compute_dtype=bnb_4bit_compute_dtype,
                quant_type=quant_type,
            )

            new_layer.load_state_dict(child.state_dict())
            new_layer = new_layer.to(device)
            setattr(module, name, new_layer)
        else:
            quantize_4bit(child)

def quantize_8bit(module):
    """
    Apply 8-bit quantization to model modules using Linear8bitLt.
    Note: The compute_dtype argument has been removed as it is not accepted.
    """
    if not bnb_available:
        raise ImportError("bitsandbytes is required for 8-bit quantization")
        
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            device = child.weight.data.device

            has_bias = child.bias is not None
            quant_type = "int8"

            # Use Linear8bitLt if available; note that compute_dtype is removed.
            if hasattr(bnb.nn, "Linear8bitLt"):
                new_layer = bnb.nn.Linear8bitLt(
                    in_features,
                    out_features,
                    bias=has_bias
                )
            else:
                raise AttributeError("bitsandbytes.nn does not have attribute 'Linear8bitLt'. Please ensure you have an updated version of bitsandbytes that supports 8-bit quantization.")
                
            new_layer.load_state_dict(child.state_dict())
            new_layer = new_layer.to(device)
            setattr(module, name, new_layer)
        else:
            quantize_8bit(child)

def pil_to_binary_mask(pil_image, threshold=0):
    """
    Convert PIL image to binary mask
    """
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i, j]:
                mask[i, j] = 1
    mask = (mask * 255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

def save_output_image(image, base_path="outputs", base_filename="img", seed=None):
    """
    Save output image with timestamp and seed
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_str = f"_seed{seed}" if seed is not None else ""
    filename = f"{base_filename}_{timestamp}{seed_str}.png"
    filepath = os.path.join(base_path, filename)
    
    counter = 1
    while os.path.exists(filepath):
        filename = f"{base_filename}_{timestamp}_{counter}{seed_str}.png"
        filepath = os.path.join(base_path, filename)
        counter += 1
        
    image.save(filepath)
    return filepath

def open_folder(folder_path="outputs"):
    """
    Open the folder in file explorer
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    if platform.system() == "Windows":
        os.startfile(folder_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.Popen(["open", folder_path])
    else:  # Linux
        subprocess.Popen(["xdg-open", folder_path])

def auto_crop_upload(editor_value, crop_flag):
    """
    When a user uploads an image (EditorValue) and if auto-cropping is enabled 
    (crop_flag is True) this function performs the cropping and resizing.
    It stores the original full resolution image in a global variable and updates the "auto_cropped" flag.
    """
    global ORIGINAL_IMAGE
    if editor_value is None:
        return None
    if editor_value.get("background") is None:
        return editor_value
    try:
        img = editor_value["background"].convert("RGB")
        if crop_flag:
            ORIGINAL_IMAGE = img.copy()
            print("auto_crop_upload: Original image stored with resolution:", ORIGINAL_IMAGE.size)
            width, height = img.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_img = img.crop((left, top, right, bottom))
            resized_img = cropped_img.resize((768, 1024))
            editor_value["background"] = resized_img
            if editor_value.get("layers"):
                new_layers = []
                for layer in editor_value["layers"]:
                    if layer is not None:
                        new_layer = layer.crop((left, top, right, bottom)).resize((768, 1024))
                        new_layers.append(new_layer)
                    else:
                        new_layers.append(None)
                editor_value["layers"] = new_layers
            editor_value["composite"] = resized_img
            editor_value["auto_cropped"] = True
            print("auto_crop_upload: Cropping done. Crop region:", left, top, right, bottom)
        else:
            print("auto_crop_upload: Auto crop flag disabled.")
    except Exception as e:
        print("auto_crop_upload: Error in auto crop:", e, file=sys.stderr)
    return editor_value

def start_tryon(dict, garm_img, garment_des, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed, seed, number_of_images):
    global pipe, unet, UNet_Encoder, need_restart_cpu_offloading, ORIGINAL_IMAGE

    if garm_img is None:
        raise gr.Error("Please upload a garment image.")

    print(f"start_tryon: Input dict from ImageEditor: {dict}")

    if pipe is None:
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=dtypeQuantize,
        )
        
        if load_mode == '4bit' and bnb_available:
            quantize_4bit(unet)
        elif load_mode == '8bit' and bnb_available:
            quantize_8bit(unet)

        unet.requires_grad_(False)

        tokenizer_one = AutoTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )
        noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        text_encoder_one = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            model_id,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        )

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_id,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
        )
        if load_mode == '4bit' and bnb_available:
            quantize_4bit(image_encoder)
        elif load_mode == '8bit' and bnb_available:
            quantize_8bit(image_encoder)

        if fixed_vae:
            vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=dtype)
        else:
            vae = AutoencoderKL.from_pretrained(model_id,
                                                subfolder="vae",
                                                torch_dtype=dtype,
            )

        UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            model_id,
            subfolder="unet_encoder",
            torch_dtype=dtypeQuantize,
        )

        if load_mode == '4bit' and bnb_available:
            quantize_4bit(UNet_Encoder)
        elif load_mode == '8bit' and bnb_available:
            quantize_8bit(UNet_Encoder)

        UNet_Encoder.requires_grad_(False)
        image_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)

        pipe = TryonPipeline.from_pretrained(
            model_id,
            unet=unet,
            vae=vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            scheduler=noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
        )
        pipe.unet_encoder = UNet_Encoder
        pipe.unet_encoder.to(device)
        
        if hasattr(pipe.unet, 'encoder_hid_proj') and pipe.unet.encoder_hid_proj is not None:
            pipe.unet.encoder_hid_proj.to(device)

        if load_mode == '4bit' and bnb_available:
            if pipe.text_encoder is not None:
                quantize_4bit(pipe.text_encoder)
            if pipe.text_encoder_2 is not None:
                quantize_4bit(pipe.text_encoder_2)
        elif load_mode == '8bit' and bnb_available:
            if pipe.text_encoder is not None:
                quantize_8bit(pipe.text_encoder)
            if pipe.text_encoder_2 is not None:
                quantize_8bit(pipe.text_encoder_2)
    else:
        if ENABLE_CPU_OFFLOAD:
            need_restart_cpu_offloading = True

    torch_gc()
    
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)
    
    if hasattr(pipe.unet, 'encoder_hid_proj') and pipe.unet.encoder_hid_proj is not None:
        pipe.unet.encoder_hid_proj.to(device)

    if need_restart_cpu_offloading:
        restart_cpu_offload(pipe, load_mode)
    elif ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
        if hasattr(pipe.unet, 'encoder_hid_proj') and pipe.unet.encoder_hid_proj is not None:
            pipe.unet.encoder_hid_proj.to(pipe.unet.device)

    tensor_transfrom = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    garm_img = garm_img.convert("RGB").resize((768, 1024))
    human_img_orig = dict["background"].convert("RGB")
    print("start_tryon: Received human image from editor.")

    if is_checked_crop:
        if ORIGINAL_IMAGE is not None:
            orig = ORIGINAL_IMAGE
            print("start_tryon: Using ORIGINAL_IMAGE from global with resolution:", orig.size)
        else:
            orig = human_img_orig
            print("start_tryon: GLOBAL ORIGINAL_IMAGE not found. Using auto-cropped human image with resolution:", orig.size)
        orig_w, orig_h = orig.size
        scale_factor = 1024 / orig_h
        final_w = int(orig_w * scale_factor)
        final_h = 1024
        final_background = orig.resize((final_w, final_h))
        print(f"start_tryon: Downscaled original image to final background: {final_background.size} (scale factor: {scale_factor})")
        target_width = int(min(orig_w, orig_h * (3 / 4)))
        target_height = int(min(orig_h, orig_w * (4 / 3)))
        left_orig = (orig_w - target_width) / 2
        top_orig = (orig_h - target_height) / 2
        left_final = int(left_orig * scale_factor)
        top_final = int(top_orig * scale_factor)
        crop_width_final = int(target_width * scale_factor)
        crop_height_final = int(target_height * scale_factor)
        crop_size = (crop_width_final, crop_height_final)
        print(f"start_tryon: Computed crop region on original image: left_orig: {left_orig}, top_orig: {top_orig}, target_width: {target_width}, target_height: {target_height}")
        print(f"start_tryon: Scaled crop region for final image: left_final: {left_final}, top_final: {top_final}, crop_size: {crop_size}")
        human_img = human_img_orig
    else:
        human_img = human_img_orig.resize((768, 1024))
        print("start_tryon: Auto crop not enabled, resized human image to 768x1024.")

    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', category, model_parse, keypoints)
        mask = mask.resize((768, 1024))
        print("start_tryon: Auto-masking used")
    else:
        if dict.get('layers') and len(dict['layers']) > 0 and dict['layers'][0] is not None:
            mask_layer = dict['layers'][0]
            if mask_layer.mode == "RGBA":
                mask_alpha = mask_layer.split()[-1]
            else:
                mask_alpha = mask_layer.convert("L")
            mask_alpha = mask_alpha.resize((768, 1024))
            print("start_tryon: Manual mask alpha extracted:", type(mask_alpha), mask_alpha.mode, mask_alpha.size)
            mask = pil_to_binary_mask(mask_alpha)
            print("start_tryon: Manual mask binary mask:", type(mask), mask.mode, mask.size)
        else:
            mask = Image.new('L', (768, 1024), 0)
            print("start_tryon: No manual mask provided, using default black mask")

    print("start_tryon: Mask before pipe:", type(mask), mask.mode, mask.size)

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args_apply = apply_net.create_argument_parser().parse_args(( 
        'show', 
        './configs/densepose_rcnn_R_50_FPN_s1x.yaml', 
        './ckpt/densepose/model_final_162be9.pkl', 
        'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'
    ))
    pose_img = args_apply.func(args_apply, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    if pipe.text_encoder is not None:
        pipe.text_encoder.to(device)
    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )
                    pose_img_tensor = tensor_transfrom(pose_img).unsqueeze(0).to(device, dtype)
                    garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, dtype)
                    results = []
                    current_seed = seed
                    for i in range(number_of_images):
                        if is_randomize_seed:
                            current_seed = torch.randint(0, 2**32, size=(1,)).item()
                        generator = torch.Generator(device).manual_seed(current_seed) if seed != -1 else None
                        current_seed = current_seed + i
                        
                        if hasattr(pipe.unet, 'encoder_hid_proj') and pipe.unet.encoder_hid_proj is not None:
                            pipe.unet.encoder_hid_proj.to(pipe.unet.device)
                                
                        images = pipe(
                            prompt_embeds=prompt_embeds.to(device, dtype),
                            negative_prompt_embeds=negative_prompt_embeds.to(device, dtype),
                            pooled_prompt_embeds=pooled_prompt_embeds.to(device, dtype),
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, dtype),
                            num_inference_steps=denoise_steps,
                            generator=generator,
                            strength=1.0,
                            pose_img=pose_img_tensor.to(device, dtype),
                            text_embeds_cloth=prompt_embeds_c.to(device, dtype),
                            cloth=garm_tensor.to(device, dtype),
                            mask_image=mask,
                            image=human_img,
                            height=1024,
                            width=768,
                            ip_adapter_image=garm_img.resize((768, 1024)),
                            guidance_scale=2.0,
                        )[0]
                        if is_checked_crop:
                            final_img = final_background.copy()
                            gen_img = images[0].resize(crop_size)
                            final_img.paste(gen_img, (left_final, top_final))
                            print(f"start_tryon: Pasted generated image onto final background at ({left_final}, {top_final})")
                            img_path = save_output_image(final_img, base_path="outputs", base_filename='img', seed=current_seed)
                            results.append(img_path)
                        else:
                            img_path = save_output_image(images[0], base_path="outputs", base_filename='img')
                            results.append(img_path)
                    return results, mask_gray

garm_list = os.listdir(os.path.join(example_path, "cloth"))
garm_list_path = [os.path.join(example_path, "cloth", garm) for garm in garm_list]

image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(
                sources='upload',
                type="pil",
                label='Human. Mask with pen or use auto-masking',
                interactive=True,
                height=550
            )
            imgs.upload(auto_crop_upload, inputs=[imgs, gr.Checkbox(value=True, label="Use auto-crop & resizing")], outputs=imgs)
            with gr.Row():
                category = gr.Radio(choices=["upper_body", "lower_body", "dresses"], label="Select Garment Category", value="upper_body")
                is_checked = gr.Checkbox(label="Yes", info="Use auto-generated mask (Takes 5 seconds)", value=True)
            with gr.Row():
                is_checked_crop = gr.Checkbox(label="Yes", info="Use auto-crop & resizing", value=True)
        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="pil")
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    prompt = gr.Textbox(placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts", show_label=False, elem_id="prompt")
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=8,
                examples=garm_list_path
            )
        with gr.Column():
            with gr.Row():
                masked_img = gr.Image(label="Masked image output", elem_id="masked-img", show_share_button=False)
            with gr.Row():
                btn_open_outputs = gr.Button("Open Outputs Folder")
                btn_open_outputs.click(fn=open_folder)
        with gr.Column():
            with gr.Row():
                image_gallery = gr.Gallery(label="Generated Images", show_label=True)
            with gr.Row():
                try_button = gr.Button(value="Try-on",variant='primary')
                denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=120, value=30, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)
                is_randomize_seed = gr.Checkbox(label="Randomize seed for each generated image", value=True)
                number_of_images = gr.Number(label="Number Of Images To Generate", minimum=1, maximum=9999, value=1, step=1)

    try_button.click(
        fn=start_tryon,
        inputs=[imgs, garm_img, prompt, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed, seed, number_of_images],
        outputs=[image_gallery, masked_img],
        api_name='tryon'
    )

image_blocks.launch(inbrowser=True, share=args.share)