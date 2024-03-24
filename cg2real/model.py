from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel, AutoPipelineForImage2Image
from diffusers.utils import load_image
import numpy as np
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

import cv2
from PIL import Image


def setup_pipe(low_memory):
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_8step_lora.safetensors"

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
    )
    pipe = AutoPipelineForImage2Image.from_pretrained(base, torch_dtype=torch.float16, variant="fp16", controlnet=controlnet)
    pipe.load_lora_weights(hf_hub_download(repo, ckpt))
    pipe.fuse_lora()
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe.set_ip_adapter_scale(0.6)
    if low_memory:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def edges(image):
  # get canny image
  condition = np.array(image)
  high_thresh, thresh_im = cv2.threshold(cv2.cvtColor(condition, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  lowThresh = 0.5 * high_thresh
  condition = cv2.Canny(condition, lowThresh, high_thresh)
  condition = condition[:, :, None]
  condition = np.concatenate([condition, condition, condition], axis=2)
  return Image.fromarray(condition)

def get_fidelity_params(fidelity):
    controlnet_conditioning_scale = 0.1 + fidelity * 0.4
    strength = 1 - fidelity * 0.4
    return controlnet_conditioning_scale, strength

class CG2Real():
    def __init__(self, fidelity=0.5, low_memory=False):
        self.pipe = setup_pipe(low_memory)
        self.fidelity = fidelity
    
    def __call__(self, prompt, image, iterations=3):
        l = min(image.width, image.height)
        image = image.crop((0,0,l,l)).resize((1024, 1024))
        prompt = f"professional photograph of {prompt}, 8k, best quality, highly detailed, fujifilm, shot on Nikon"
        negative_prompt = "low quality, worst quality, artwork, drawing, painting, cgi, hazy, blurry, faded"
        condition = edges(image)
        controlnet_conditioning_scale, strength = get_fidelity_params(self.fidelity)
        res = image
        for i in range(iterations):
            res = self.pipe(
                        prompt,
                        strength=strength,
                        image=res,
                        ip_adapter_image=image, 
                        control_image=condition, 
                        num_inference_steps=8, 
                        guidance_scale=0,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                    ).images[0]
        return res