"""
Large Multimodal Models (LMM) Tools

V (Image + Video), T, A

directly generate

V U - Qwen2.5 VL G - (Stable diffusion) + refiner, or generated video. 

T U & G Qwen2.5 (LLM) + extend + refiner.

A U - SeaLLMs-Audio (Built upon Qwen2.5 7B) Audio, G - (Suno Bark)
"""
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoProcessor, AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2AudioForConditionalGeneration, BarkModel, BarkProcessor
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableAudioPipeline, FluxPipeline, AudioLDM2Pipeline, StableDiffusion3Pipeline
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor



def load_model(model_type: str, model_name: str, device: str):
    max_memory_each_gpu = "46GiB"
    max_memory_dict = {0: "0GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB", 4: "0GiB", 5: "0GiB", 6: "0GiB", 7: "0GiB"}
    if model_type == 'image_understanding':
        
        if model_name == 'Qwen/Qwen2.5-VL-32B-Instruct':
            gpu_ids = device.split(",")
            for i in gpu_ids:
                max_memory_dict[int(i)] = max_memory_each_gpu
                
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            max_memory=max_memory_dict,)
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                # "Qwen/Qwen2.5-VL-32B-Instruct",
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={'': device}
            ).eval()
        processor = AutoProcessor.from_pretrained(model_name)
        return {'model': model, 'processor': processor}
    elif model_type == 'image_generator_flux1':
        # "black-forest-labs/FLUX.1-dev"
        model = FluxPipeline.from_pretrained(model_name, 
                                            torch_dtype=torch.bfloat16).to(device)
        # model.enable_model_cpu_offload(device=device)
        return {'model': model, 'processor': None}
    elif model_type == 'image_generator_sd3':
        # model_id = "stabilityai/stable-diffusion-3.5-large"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_name,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
        )
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_name, 
            transformer=model_nf4,
            torch_dtype=torch.bfloat16,
            
        )
        # pipeline.enable_model_cpu_offload(device=device)
        pipeline.to(device)
        
        return {'model': pipeline, 'processor': None}
    elif model_type == 'image_generation_refiner':
        pass
    elif model_type == 'text_understanding':
        model = AutoModelForCausalLM.from_pretrained(
            # "Qwen/Qwen2.5-7B-Instruct-1M",
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={'': device}
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return {'model': model, 'processor': tokenizer}
    elif model_type == 'audio_understanding':
        # audio_understanding = 'mispeech/r1-aqa'
        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name, 
                                                                device_map={'': device}, 
                                                                torch_dtype=torch.bfloat16)
        return {'model': model, 'processor': processor}
    elif model_type == 'audio_generator_bark':
        # audio_generation = 'suno/bark'
        processor = BarkProcessor.from_pretrained(model_name)
        model = BarkModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
        return {'model': model, 'processor': processor}
    elif model_type == 'audio_generator_ldm2':
        # repo_id = "cvssp/audioldm2-large"
        pipe = AudioLDM2Pipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        pipe = pipe.to(device)
        return {'model': pipe, 'processor': None}
    elif model_type == 'audio_generator_sd1':
        # "stabilityai/stable-audio-open-1.0"
        model = StableAudioPipeline.from_pretrained(model_name, 
                                                    torch_dtype=torch.float16)
        model.to(device)
        return {'model': model, 'processor': None}
    elif model_type == 'omni_understanding':
        model = Qwen2_5OmniModel.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map={'': device},
            attn_implementation="flash_attention_2",
            enable_audio_output=False,
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
        return {'model': model, 'processor': processor}
    else:
        raise ValueError(f"Unrecognized model name: {model_type}")
    
    
if __name__ == "__main__":
    pass
    # device = "cuda:4" if torch.cuda.is_available() else "cpu"
    
    
    # load image generation model
    # model_type = 'image_generation'
    # model_info = load_model(model_type, device)
    # print(f"Loaded {model_type} model on {device}")
    # print(model_info['model'])
    
    # prompt = "A whimsical and creative image depicting a hybrid creature that is a mix of a waffle and a hippopotamus, basking in a river of melted butter amidst a breakfast-themed landscape. It features the distinctive, bulky body shape of a hippo. However, instead of the usual grey skin, the creature's body resembles a golden-brown, crispy waffle fresh off the griddle. The skin is textured with the familiar grid pattern of a waffle, each square filled with a glistening sheen of syrup. The environment combines the natural habitat of a hippo with elements of a breakfast table setting, a river of warm, melted butter, with oversized utensils or plates peeking out from the lush, pancake-like foliage in the background, a towering pepper mill standing in for a tree.  As the sun rises in this fantastical world, it casts a warm, buttery glow over the scene. The creature, content in its butter river, lets out a yawn. Nearby, a flock of birds take flight"

    # image = model_info['model'](
    #     prompt=prompt,
    #     num_inference_steps=28,
    #     guidance_scale=4.5,
    #     max_sequence_length=512,
    #     smart_split=True
    # ).images[0]
    # image.save("whimsical_custom.png")
    
    
    # model_type = 'image_understanding'
    # model_info = load_model(model_type, device)
    
    # print(f"Loaded {model_type} model on {device}")
    
    
    # messages = [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image", "image": "file:///mnt/share/keguanzhou/projects/know-bridger2/whimsical_custom.png"},
    #                     {"type": "text", "text": "Describe this image."},
    #                 ],
    #             }
    #         ]
    # processor = model_info['processor']
    # model = model_info['model']
    # from qwen_vl_utils import process_vision_info
    # # Preparation for inference
    # text = processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    # image_inputs, video_inputs = process_vision_info(messages)
    # inputs = processor(
    #     text=[text],
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )
    # inputs = inputs.to(device)

    # # Inference: Generation of the output
    # generated_ids = model.generate(**inputs, max_new_tokens=128)
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # print(output_text)
    
    # from metrics import metrics_language
    # preds = output_text[0]
    # target = prompt
    # result = metrics_language(preds, target, device=device)
    # print(result)
    
    
    # load text understanding model
    # model_type = 'text_understanding'
    # model_info = load_model(model_type, device)
    # print(f"Loaded {model_type} model on {device}")
    # model = model_info['model']
    # tokenizer = model_info['processor']
    # prompt = "Give me a short introduction to large language model."
    # messages = [
    #     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    #     {"role": "user", "content": prompt}
    # ]
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # generated_ids = model.generate(
    #     **model_inputs,
    #     max_new_tokens=512
    # )
    # generated_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]

    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(response)



    