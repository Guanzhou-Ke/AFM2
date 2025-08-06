import os
import json
from typing import Dict
import soundfile as sf

from afm2.prompts.generation import SYSTEM_PROMPT, GENERATION_AUDIO_PROMPT, GENERATION_AUDIO_PROMPT_WITHOUT_GRAPH
from afm2.schema import Message
from afm2.logger import logger
from afm2.llm_client import LLM
from afm2.utils import extract_json_from_response, generate_tool_calling



class AudioGeneration:
    """
    A versatile multimodal agent that uses planning and tools to complete the missing modality.

    
    """
    def __init__(self, llm: LLM, generator_api, name='AudioGeneration'):
        self.name = name
        self.llm = llm
        self.generator_api = generator_api
        self.system_prompt: str = SYSTEM_PROMPT
    
    
    def _process_json(self, response: str):
        # prompts_str = response.split('```')[1].strip()[4:].replace('\n', '')
        prompts = extract_json_from_response(response)
        # check all the key name is `prompts`
        candidates = []
        for item in prompts['candidates']:
            if 'prompts' in item:
                candidates.append(item)
        prompts['candidates'] = candidates
        return prompts
    
    def _save_results(self, results: Dict, work_dir: str):
        """Save the results to a file."""
        file_path = os.path.join(work_dir, 'completion/audio/prompt.json')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {file_path}")
    
    
    async def step(self, summaries, work_dir='./default_dir', use_rag=False, feedbacks=None):
        """Process current state and decide next actions with appropriate context."""
        system_prompt = self.system_prompt.format(target='audio')
        text_info = summaries['text_understanding'] if 'text_understanding' in summaries else "None"
        image_info = summaries['image_understanding'] if 'image_understanding' in summaries else "None"
        if use_rag:
            guess = summaries['knowledge_extractor']
            next_prompt = GENERATION_AUDIO_PROMPT.format(text_info=text_info, image_info=image_info, guess=guess)
        else:
            next_prompt = GENERATION_AUDIO_PROMPT_WITHOUT_GRAPH.format(text_info=text_info, image_info=image_info)
        messages = [Message.system_message(system_prompt),
                  Message.user_message(next_prompt)]
        response = await self.llm.ask(messages=messages)
        json_response = self._process_json(response)
        # Generate images based on the prompt
        os.makedirs(os.path.join(work_dir, 'completion/audio'), exist_ok=True)
        logger.info(f"Generating audios.")
        for i, item in enumerate(json_response['candidates']):
            prompt_text = item['prompts']
            json_response['candidates'][i]['generations'] = []
            response = await generate_tool_calling(self.generator_api, prompt_text, work_dir, f'prompt_{i}', timeout=600)
            if isinstance(response, str):
                # If the response is a string, it might be a JSON string
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON from response: {response}")
                    continue             
            json_response['candidates'][i]['generations'].append(response['output_path'])                  
        self._save_results(json_response, work_dir)
        return response

