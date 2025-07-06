import gradio as gr

import os 
import time 
import asyncio

from pipeline import PromptEnhancer


async def advancedPromptPipeline(InputPrompt, model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.0):
    
    if model == "gpt-4o":
        i_cost=5/10**6
        o_cost=15/10**6
    elif model == "meta-llama/llama-4-maverick-17b-128e-instruct":
        i_cost=0.15/10**6
        o_cost=0.6/10**6
    
    enhancer = PromptEnhancer(model, temperature)
    
    start_time = time.time()
    advanced_prompt = await enhancer.enhance_prompt(InputPrompt, perform_eval=False)
    elapsed_time = time.time() - start_time


    """return {
        "model": model,
        "elapsed_time": elapsed_time,
        "prompt_tokens": enhancer.prompt_tokens,
        "completion_tokens": enhancer.completion_tokens,
        "approximate_cost": (enhancer.prompt_tokens*i_cost)+(enhancer.completion_tokens*o_cost),
        "inout_prompt": input_prompt,
        "advanced_prompt": advanced_prompt["advanced_prompt"],
    }"""

    return advanced_prompt["advanced_prompt"]


demo = gr.Interface(fn=advancedPromptPipeline, 
                    inputs=[
                        gr.Textbox(lines=11, placeholder="Enter your prompt", label="Input Prompt", min_width=100),
                        gr.Radio(["meta-llama/llama-4-maverick-17b-128e-instruct", "gpt-4o"], value="meta-llama/llama-4-maverick-17b-128e-instruct", label="Select Model", info="Recommended: meta-llama/llama-4-maverick-17b-128e-instruct"),
                        gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Temperature", info="Recommended: Temperature=0.0")
                        ], 
                    outputs=[
                        gr.Textbox(lines=23, label="Advanced Prompt", show_copy_button=True, autoscroll=False, min_width=220),
                        ],
                    title="Advanced Prompt Generator",
                    description="This tool will enhance any given input for the optimal output!",
                    theme="Base",
                   )


if __name__ == "__main__":
    demo.launch()