from PIL import Image
import torch
from grn_pipeline import GRNPipeline

# load pipeline
pipeline = GRNPipeline.from_pretrained(hf_repo_id='bytedance-research/GRN', device='cpu')
pipeline = pipeline.to('cuda')

# generatie one image
result = pipeline(
    prompt="A cute cat playing in the garden",
    guidance_scale=3.0,
    temperature=1.1,
    num_inference_steps=50,
    width=1024,
    height=1024,
    content_type='image',
    seed=42
)
image = result.images[0]
image.save('./generated_image.jpg')
