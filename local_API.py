import torch
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    AutoPipelineForText2Image,
    StableDiffusion3Pipeline,
    BitsAndBytesConfig,
    SD3Transformer2DModel
)
from transformers import T5EncoderModel, CLIPTokenizer
# Load tokenizer just once
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def truncate_prompt(prompt:str, max_tokens: int=77):
    tokens = clip_tokenizer(prompt, truncation=True, max_length=max_tokens, return_tensors="pt")
    decoded = clip_tokenizer.batch_decode(tokens["input_ids"], skip_special_tokens=True)[0]
    return decoded

# -------------------------------------
# 1) Stable Diffusion v1.5 (half-precision)
# -------------------------------------
# Load the Stable Diffusion v1.5 model
pipe_sd15 = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# # -------------------------------------
# # 2) Flux + Ghibli LoRA (bf16)
# # -------------------------------------
# pipe_flux = AutoPipelineForText2Image.from_pretrained(
#     "black-forest-labs/FLUX.1-dev",
#     torch_dtype=torch.bfloat16
# ).to("cuda")
# pipe_flux.load_lora_weights(
#     "openfree/flux-chatgpt-ghibli-lora",
#     weight_name="flux-chatgpt-ghibli-lora.safetensors"
# )

# # -------------------------------------
# # 3) Stable Diffusion 3.5 (4-bit NF4 quant + CPU offload)
# # -------------------------------------
# model_id_sd3 = "stabilityai/stable-diffusion-3.5-large-turbo"

# # 4-bit NF4 config
# nf4_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# # load quantized transformer
# transformer_nf4 = SD3Transformer2DModel.from_pretrained(
#     model_id_sd3,
#     subfolder="transformer",
#     quantization_config=nf4_config,
#     torch_dtype=torch.bfloat16
# )

# # load quantized text encoder
# t5_nf4 = T5EncoderModel.from_pretrained(
#     "diffusers/t5-nf4",
#     torch_dtype=torch.bfloat16,
# )

# # build SD3 pipeline with quantized modules
# pipe_sd3 = StableDiffusion3Pipeline.from_pretrained(
#     model_id_sd3,
#     transformer=transformer_nf4,
#     text_encoder_3=t5_nf4,
#     torch_dtype=torch.bfloat16
# )
# # offload to CPU when not actively used
# pipe_sd3.enable_model_cpu_offload()

# -------------------------------------
# Dispatcher
# -------------------------------------
def generate_image(prompt: str, model_name: str) -> Image.Image:
    if model_name == "stable-diffusion-v1-5/stable-diffusion-v1-5":
        safe_prompt = truncate_prompt(prompt)
        if not safe_prompt.strip():
            raise ValueError("Prompt is empty after truncation")
        print(f"[DEBUG] Generating image for: {safe_prompt}")
        return pipe_sd15(prompt=safe_prompt).images[0]

    # elif model_name == "openfree/flux-chatgpt-ghibli-lora":
    #     return pipe_flux(prompt).images[0]

    # elif model_name == "stabilityai/stable-diffusion-3.5-large-turbo":
    #     return pipe_sd3(
    #         prompt=prompt,
    #         num_inference_steps=4,
    #         guidance_scale=0.0,
    #         max_sequence_length=512
    #     ).images[0]

    else:
        raise ValueError(f"Unsupported local model: {model_name}")
