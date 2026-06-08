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

prompt="The man, of medium build with short, dark, curly hair, stands centered in the frame, wearing a simple white t-shirt that contrasts with the greenery behind him. He holds a dark smartphone, likely a modern model with a triple-lens camera setup, in his right hand, angled slightly toward his body. His gaze is fixed on the screen, and his facial expression shifts subtly\u2014smiling, nodding, and occasionally pursing his lips\u2014as if reacting to content on the phone. The background features a mix of tall green trees and shrubs, with a light blue metal fence running horizontally across the mid-ground, suggesting a garden or rural boundary. The overcast sky diffuses the light, creating soft shadows and a calm, neutral atmosphere. The man\u2019s slight head movements and micro-expressions indicate engagement, possibly reading or responding to a message or video. The composition places him as the focal point, with the natural, slightly blurred background reinforcing his isolation in the moment. The relative stillness of the scene, apart from his subtle gestures, suggests a private, introspective interaction with technology in a serene outdoor setting"

# Generate one video
result = pipeline(
    prompt=f"{prompt}. masterpiece, high quality.",
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
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
