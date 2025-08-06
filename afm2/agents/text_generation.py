import json
import os
from typing import Dict


from afm2.prompts.generation import (SYSTEM_PROMPT, 
                                    GENERATION_TEXT_PROMPT, 
                                    GENERATION_TEXT_PROMPT_WITHOUT_GRAPH,
                                    GENERATION_TEXT_PROMPT_REFINED)
from afm2.schema import Message
from afm2.logger import logger
from afm2.llm_client import LLM
from afm2.utils import load_config, extract_json_from_response


class TextGeneration:
    """
    A versatile multimodal agent that uses planning and tools to complete the missing modality.

    
    """
    
    def __init__(self, llm: LLM, ):
        self.llm = llm
        self.system_prompt: str = SYSTEM_PROMPT
        self.name = "text_generation"

    
    def _process_json(self, response: str):
        # prompts_str = response.split('```')[1].strip()[4:].replace('\n', '')
        prompts = extract_json_from_response(response)
        # check all the key name is `prompts`
        candidates = []
        for item in prompts['candidates']:
            if 'text' in item:
                candidates.append(item)
        prompts['candidates'] = candidates
        return prompts
    
    def _save_results(self, results: Dict, work_dir: str):
        """Save the results to a file."""
        file_path = os.path.join(work_dir, 'completion/text/prompt.json')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {file_path}")
    
    async def step(self, summaries, work_dir='./default_dir', use_rag=False, feedbacks=None):
        """Process current state and decide next actions with appropriate context."""
        if feedbacks is None:
            system_prompt = self.system_prompt.format(target='text')
            image_info = summaries['image_understanding'] if 'image_understanding' in summaries else "None"
            audio_info = summaries['audio_understanding'] if 'audio_understanding' in summaries else "None"
            if use_rag:
                guess = summaries['knowledge_extractor']
                next_prompt = GENERATION_TEXT_PROMPT.format(image_info=image_info, audio_info=audio_info, guess=guess)
            else:
                next_prompt = GENERATION_TEXT_PROMPT_WITHOUT_GRAPH.format(image_info=image_info, audio_info=audio_info)            
            messages = [Message.system_message(system_prompt),
                    Message.user_message(next_prompt)]
            response = await self.llm.ask(messages=messages)
            json_response = self._process_json(response)
        else:
            # we need to integrate the feedback into the prompt
            json_response = {}
            json_response['candidates'] = []
            for original_prompt in feedbacks['candidates']:
                prompt = f"The original text is: {original_prompt['text']}."
                next_prompt = GENERATION_TEXT_PROMPT_REFINED.format(image_info=image_info, audio_info=audio_info)
                messages = [Message.system_message(GENERATION_TEXT_PROMPT_REFINED),
                        Message.user_message(prompt)]
                response = await self.llm.ask(messages=messages)
                refined_prompt = response
                json_response['candidates'].append({'text': refined_prompt})
            logger.info(f"Refined prompt: {refined_prompt}")
            
        self._save_results(json_response, work_dir)
        return response

