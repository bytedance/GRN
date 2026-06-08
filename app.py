import os
import sys
import traceback
import torch
import gradio as gr
import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grn_pipeline import GRNPipeline

# Global pipeline
pipe = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_pipeline():
    global pipe
    print(f"Loading GRN pipeline ({device=})...")
    # 从 Hugging Face Hub 下载权重
    pipe = GRNPipeline.from_pretrained(
        hf_repo_id='bytedance-research/GRN',
        task='T2I',
        pn='1M', 
        model='GRN2b',
        use_slow_attn=True,
        device=device,
    )
    print("Pipeline loaded successfully!")
    return pipe

# @spaces.GPU #[uncomment to use ZeroGPU]
@spaces.GPU(duration=40)
def generate(prompt, content_type="image", guidance_scale=3.0, temperature=1.0, seed=42, width=1024, height=1024):
    global pipe
    if pipe is None:
        try:
            pipe = load_pipeline()
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            traceback.print_exc()
            return f"Error loading pipeline: {e}\n\n{traceback.format_exc()}"
    
    try:
        result = pipe(
            prompt="<T2I>"+prompt,
            guidance_scale=guidance_scale,
            temperature=temperature,
            complexity_aware_Tmin=10,
            complexity_aware_Tmax=50,
            complexity_aware_k = 0,
            complexity_aware_b = 50,
            complexity_aware_wp = 5,
            snr_shift = 1.,
            h_div_w=1.,
            content_type=content_type,
            seed=seed,
            width=width,
            height=height
        )
        
        if content_type == "image" and hasattr(result, 'images'):
            return result.images[0]
        elif content_type == "video" and hasattr(result, 'videos'):
            return result.videos[0]
        return f"Error: Invalid result from pipeline"
    except Exception as e:
        print(f"Error generating content: {e}")
        traceback.print_exc()
        return f"Error generating content: {e}\n\n{traceback.format_exc()}"

def create_demo():
    with gr.Blocks(title="GRN: Generative Refinement Networks", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# GRN: Generative Refinement Networks")
        gr.Markdown("Text-to-Image generation using GRN")
        
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Enter your prompt here...",
                    value="A cute cat playing in the garden"
                )
                
                content_type = gr.Radio(
                    choices=["image"], # , "video"
                    value="image",
                    label="Content Type"
                )
                
                with gr.Accordion("Settings", open=True):
                    guidance_scale = gr.Slider(minimum=0, maximum=10, value=3.0, label="Guidance Scale")
                    temperature = gr.Slider(minimum=0.1, maximum=1.5, value=1.1, label="Temperature")
                    seed = gr.Number(value=42, label="Seed", precision=0)
                    width = gr.Number(value=1024, label="Width", precision=0)
                    height = gr.Number(value=1024, label="Height", precision=0)
                
                generate_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column():
                output = gr.Gallery(label="Output", show_label=True, elem_id="gallery", columns=1, height="auto", preview=True, object_fit="contain")
        
        def generate_and_display(prompt, content_type, guidance_scale, temperature, seed, width, height):
            result = generate(prompt, content_type, guidance_scale, temperature, seed, width, height)
            if result:
                return [result]
            return []
        
        generate_btn.click(
            fn=generate_and_display,
            inputs=[prompt_input, content_type, guidance_scale, temperature, seed, width, height],
            outputs=output
        )
        
        gr.Examples(
            examples=[
                ["A majestic lion standing on a cliff at sunset", "image", 3.0, 1.0, 42, 1024, 1024],
            ],
            inputs=[prompt_input, content_type, guidance_scale, temperature, seed, width, height],
            cache_examples=False
        )
    
    return demo

if __name__ == "__main__":
    try:
        load_pipeline()
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        traceback.print_exc()
    
    demo = create_demo()
    demo.launch()