from PIL import Image
from grn_pipeline import GRNPipeline

# Load pipeline
pipeline = GRNPipeline.from_pretrained(
    hf_repo_id='bytedance-research/GRN',
    task='T2I',
    pn='1M', 
    model='GRN2b',
    use_slow_attn=True,
    device='cpu',
).to('cuda')

# Generate one image
result = pipeline(
    prompt="<T2I>" + "A cute cat playing in the garden",
    guidance_scale=3.0,
    temperature=1.1,
    complexity_aware_Tmin=10,
    complexity_aware_Tmax=50,
    complexity_aware_k = 0,
    complexity_aware_b = 50,
    complexity_aware_wp = 5,
    snr_shift = 1.,
    h_div_w=1.,
    content_type='image',
    seed=42,
)
image = result.images[0]
image.save('./generated_image.jpg')
