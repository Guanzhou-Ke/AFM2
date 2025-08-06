import os
import argparse
import asyncio
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import gc
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import soundfile as sf

from afm2.servers.lmm_tools import load_model
from afm2.utils import load_config

image_model_lock = asyncio.Lock()
audio_model_lock = asyncio.Lock()


class GenerationRequest(BaseModel):
    prompt: str
    output_path: str
    save_name: str


app = FastAPI(title="Async Generator Agent API")

IMAGE_CONFIG = {}
AUDIO_CONFIG = {}
image_pipe: Optional[Any] = None
audio_pipe: Optional[Any] = None


@app.on_event("startup")
def load_config_and_model():
    global IMAGE_CONFIG, AUDIO_CONFIG, image_pipe, audio_pipe

    config = load_config("servers/generator_config.yaml")
    
    if "image" in config.default.infer_type:
        IMAGE_CONFIG = config.image_generator
        print(f"🚀 Loading generator {IMAGE_CONFIG.model_name} on {IMAGE_CONFIG.device}...")
        mode_info = load_model(IMAGE_CONFIG.model_type, IMAGE_CONFIG.model_name, IMAGE_CONFIG.device)
        image_pipe = mode_info['model']
        processor = mode_info['processor']
        print("✅ Model ready.")
        
    if "audio" in config.default.infer_type:
        AUDIO_CONFIG = config.audio_generator
        print(f"🚀 Loading generator {AUDIO_CONFIG.model_name} on {AUDIO_CONFIG.device}...")
        mode_info = load_model(AUDIO_CONFIG.model_type, AUDIO_CONFIG.model_name, AUDIO_CONFIG.device)
        audio_pipe = mode_info['model']
        processor = mode_info['processor']
        print("✅ Model ready.")
    
        
        
@app.post("/audio_generate")
async def audio_generate(request: GenerationRequest):
     # Acquire the lock to ensure thread safety
    async with audio_model_lock:
        global AUDIO_CONFIG, audio_pipe

        if audio_pipe is None:
            return {"error": "Model is not loaded."}


        loop = asyncio.get_event_loop()

        def run_generation():
            audios = audio_pipe(
                request.prompt,
                negative_prompt=AUDIO_CONFIG.negative_prompt,
                num_inference_steps=AUDIO_CONFIG.num_inference_steps,
                audio_end_in_s=AUDIO_CONFIG.audio_end_in_s,
                num_waveforms_per_prompt=AUDIO_CONFIG.num_waveforms_per_prompt,
            ).audios
            return audios

        audios = await loop.run_in_executor(None, run_generation)

        os.makedirs(os.path.join(request.output_path, 'completion/audio'), exist_ok=True)
        for i, audio in enumerate(audios):
            output = audio.T.float().cpu().numpy()
            sf.write(os.path.join(request.output_path, f'completion/audio/{request.save_name}_{i}.wav'), output, 
                         audio_pipe.vae.sampling_rate)
            
        del audios
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "message": "Generation completed.",
            "prompt": request.prompt,
            "output_path": str(os.path.join(request.output_path, f'completion/audio/{request.save_name}_{i}.wav'))
        }
        
        
@app.post("/image_generate")
async def image_generate(request: GenerationRequest):
    # Acquire the lock to ensure thread safety
    async with image_model_lock:
        global IMAGE_CONFIG, image_pipe

        if image_pipe is None:
            return {"error": "Model is not loaded."}


        loop = asyncio.get_event_loop()

        def run_generation():
            with torch.inference_mode():
                    images = image_pipe(
                    prompt=request.prompt,
                    height=IMAGE_CONFIG.height,
                    width=IMAGE_CONFIG.width,
                    negative_prompt=IMAGE_CONFIG.negative_prompt,
                    num_inference_steps=IMAGE_CONFIG.num_inference_steps,
                    guidance_scale=IMAGE_CONFIG.guidance_scale,
                    max_sequence_length=IMAGE_CONFIG.max_sequence_length,
                    num_images_per_prompt=IMAGE_CONFIG.num_images_per_prompt,
                ).images
            return images

        images = await loop.run_in_executor(None, run_generation)

        os.makedirs(os.path.join(request.output_path, 'completion/image'), exist_ok=True)
        for i, image in enumerate(images):
            image.save(os.path.join(request.output_path, f'completion/image/{request.save_name}_{i}.png'))
            
        del images
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "message": "Generation completed.",
            "prompt": request.prompt,
            "output_path": str(os.path.join(request.output_path, f'completion/image/{request.save_name}_{i}.png'))
        }

def run_server(port: int):
    uvicorn.run("generator_server:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(args.port)
