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
    model='GRN8b', # 'GRN2b' or 'GRN8b'
    use_slow_attn=False,
    device='cpu'
).to('cuda')

first_frame_path='./assets/i2v_example.jpg'
# support English and Chinese prompt, GRN prefers longer and detailed prompt
prompt='视频展示了一辆红色敞篷跑车在城市道路中行驶的连续画面。车辆以中等速度前进，车身光滑，反射着黄昏的暖光，黑色轮毂与红色车漆形成对比。驾驶员为男性，专注地操控方向盘，姿态放松。道路两侧排列着高大的棕榈树，背景中可见石质围栏和模糊的建筑轮廓。随着视频推进，一辆白色SUV从后方快速驶过，产生动态模糊，突显跑车的稳定行驶。镜头保持相对固定的侧前方视角，轻微跟随车辆移动，捕捉车身线条与光影变化。整体画面色调温暖，光线柔和，营造出一种优雅而动感的都市驾驶氛围。'
# Generate one video
result = pipeline(
    prompt=f"<I2V>{prompt} high aesthetic and high quality video.",
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
    duration=2., # 2~5s
    first_frame_condition=True,
    first_frame_path=first_frame_path,
    content_type='video',
    seed=42,
)
video_file = result.videos[0]
