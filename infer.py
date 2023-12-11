from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionXLInpaintPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16")

prompt2 = "landscape of the beautiful city of paris rebuilt near the pacific ocean in sunny california, amazing weather, sandy beach, palm trees, splendid haussmann architecture, digital painting, highly detailed, intricate, without duplication, art by craig mullins, greg rutkwowski, concept art, matte painting, trending on artstation"
n_prompt="duplicate, ugly, tiling, poorly drawn hands, poorly drawn fingers, disconnected feet, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, awkward face shapes, unwanted open mouth, broken text"
image = pipe(prompt=prompt2,num_inference_steps=1, guidance_scale=0.0, negative_prompt=n_prompt).images[0]
image_name = "t1_KSI.png"
image.save(image_name)
print("Generated image saved to:",image_name)