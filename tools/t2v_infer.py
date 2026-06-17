from tools.grn_pipeline import GRNPipeline

negative_prompt = (
    # --- quality ---
    "ugly, blurry, low-resolution, low-detail, low-quality, noisy, grainy, "
    "overexposed, underexposed, oversaturated, undersaturated, soft focus, "
    "artifacts, compression artifacts, jpeg artifacts, flickering, "
    # --- style ---
    "painting, oil painting, illustration, drawing, sketch, cartoon, anime, manga, "
    "3d, cgi, render, digital art, "
    "plastic, waxy, glossy, fake, unnatural, "
    # --- skin, figure ---
    "plastic skin, waxy skin, over-smoothed skin, doll-like, "
    "deformed, mutated, disfigured, bad anatomy, bad hands, extra fingers, missing fingers, extra limbs, "
    # --- motion ---
    "still image, static, motionless, frozen, "
    "unnatural motion, reversed motion, stuttering, choppy, "
    # --- misc ---
    "text, watermark, logo, signature, username, "
    "crowded background, bad composition"
)

# Load pipeline
pipeline = GRNPipeline.from_pretrained(
    hf_repo_id='bytedance-research/GRN', 
    task='T2V', 
    pn='0.41M', 
    model='GRN2b',
    use_slow_attn=False,
    device='cpu'
).to('cuda')

prompt="The video captures a male performer on stage, wearing a black cap, black t-shirt, and a black beaded bracelet on his left wrist, with a tattoo visible on his left forearm. He holds a microphone close to his mouth with his left hand while raising his right arm in a dynamic gesture, suggesting energetic performance. The stage is illuminated with intense blue and purple lighting, creating a moody atmosphere; a focused spotlight beam is visible in the background, adding depth. The performer’s facial expression is intense, eyes closed or squinting, indicating emotional engagement. The camera maintains a close-up, slightly angled shot of his upper body, with minimal movement, emphasizing his actions and expressions. Across the frames, the lighting subtly shifts, enhancing the visual dynamics of the performance without altering the scene’s core composition"
prompt='一个头戴耳机的男人正在一个工作室里对着麦克风唱歌，偶尔晃动头部'
# Generate one video
result = pipeline(
    prompt=f"<T2V>{prompt}. The quality is very high!",
    negative_prompt=negative_prompt,
    guidance_scale=3.0,
    temperature=1.0,
    complexity_aware_Tmin=10,
    complexity_aware_Tmax=50,
    complexity_aware_k = 0,
    complexity_aware_b = 50,
    complexity_aware_wp = 5,
    snr_shift = 1.,
    h_div_w=9/16,
    duration=2.,
    first_frame_condition=False,
    content_type='video',
    seed=42,
)
video_file = result.videos[0]
