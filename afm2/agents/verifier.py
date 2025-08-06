import os
from typing import Dict
import json
from collections import defaultdict


from afm2.prompts.verifier import VERIFICATION_TEXT_PROMPT, VERIFICATION_IMAGE_PROMPT, VERIFICATION_AUDIO_PROMPT
from afm2.logger import logger
from afm2.llm_client import OmniClient
from afm2.utils import load_config, load_json_from_file, save_json, extract_json_from_response
from afm2.schema import Message


def convert_defaultdict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict(v) for k, v in d.items()}
    return d  

class Verifier:
    """
    A versatile multimodal agent that uses planning and tools to complete the missing modality.

    
    """
    
    def __init__(self, omniclient: OmniClient, 
                 penalty_score=0.1, 
                 generation_threshold=4, 
                 understanding_threshold=4, 
                 max_generation_verification_step=3,
                 only_verify_generation=True):
        
        self.only_verify_generation = only_verify_generation
        self.omniclient = omniclient
        self.penalty_score = penalty_score
        self.generation_threshold = generation_threshold
        self.understanding_threshold = understanding_threshold
        self.max_generation_verification_step = max_generation_verification_step
        self.current_generation_step = 0
        

    
    
    def _remove_file_prefix(self, file_path: str) -> str:
        """
        Remove the prefix from the file path.
        """
        if file_path.startswith('file://'):
            file_path = file_path[7:]
        return file_path
    
    def _process_response(self, response):
        """
        Process the response from the LLM.
        """
        
        score_str = response.split('```')[1].strip()[4:].replace('\n', '')
        scores = json.loads(score_str)
        return scores
        
    
    async def _verify_generation_text(self, image=None, audio=None, generated_texts=None, judge_model=None):
        """
        Verify the generation text using the LLM.
        """
        result = defaultdict(dict)
        feedbacks = {
            'candidates': [],
        }
        image = self._remove_file_prefix(image) if image else None
        audio = self._remove_file_prefix(audio) if audio else None
        if image is None and audio is None:
            raise ValueError("Both image and audio are None.")
        best_score = 0.
        best_text = None
        result['best_score'] = best_score
        need_to_regenerate = False
        for text in generated_texts:
            # score2image = self._calculate_similarity(image=image, text=text)
            # score2audio = self._calculate_similarity(audio=audio, text=text)
            feedbacks['candidates'].append({'text': text})
            overall_score = 0.
            if image is not None:
                message = [
                    {"role": "user",
                    "content": [
                        {"type": "text", "text": f"Judge the following generated text with respect to the image.\n Generated Text: {text}"},
                        {"type": "image", "image": image},
                    ]}
                ]
                message = Message.system_message(VERIFICATION_TEXT_PROMPT) + message
                response_it = await self.omniclient.ask(messages=message)
                logger.info(f"Response to image: {response_it}")
                score_it = extract_json_from_response(response_it)
                result[text]['it'] = score_it
                overall_score += (score_it['overall_score'] - min((score_it['hallucinated_assertions'] * self.penalty_score), 1))
            
            if audio is not None:
                message = [
                    {"role": "user",
                    "content": [
                        {"type": "text", "text": f"Judge the following generated text with respect to the audio.\n Generated Text: {text}"},
                        {"type": "audio", "audio": audio},
                    ]}
                ]
                message = Message.system_message(VERIFICATION_TEXT_PROMPT) + message
                response_at = await self.omniclient.ask(messages=message)
                logger.info(f"Response to audio: {response_at}")
                
                score_at = extract_json_from_response(response_at)
                result[text]['at'] = score_at
                overall_score += (score_at['overall_score'] - min((score_it['hallucinated_assertions'] * self.penalty_score), 1))
                
            if image is not None and audio is not None:
                # for normalize the overscore.
                overall_score /= 2
            
            result[text]['overall_score'] = overall_score
            
            if overall_score > best_score:
                # update the best score and text
                best_score = overall_score
                best_text = text
                result['best_text'] = best_text
                result['best_score'] = best_score
        return convert_defaultdict(result), need_to_regenerate, feedbacks
                 
    
    async def _verify_generation_image(self, audio=None, text=None, generated_images=None, judge_model=None, base_path=None):    
        """
        Verify the generation image using the LMM.
        """
        result = defaultdict(dict)
        feedbacks = defaultdict(dict)
        audio = self._remove_file_prefix(audio) if audio else None
        best_score = 0.
        best_prompt = None
        best_image = None
        result['best_score'] = best_score
        need_to_regenerate = False
        for item in generated_images:
            generated_promt, image_paths = item['prompts'], item['generations']
            for image_path in image_paths:
                if text is not None:
                # 1. judge the image with respect to the text
                    prompt = f"Judge the following generated image with respect to the generated prompt and ground-truth text.\n Generated Prompt: {generated_promt}.\n Ground-truth Text: {text}."
                    overall_score = 0.
                    message = [
                        {"role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image": os.path.join(base_path, image_path)},
                        ]}
                    ]
                    
                    message = Message.system_message(VERIFICATION_IMAGE_PROMPT) + message
                    response_it = await self.omniclient.ask(messages=message)
                    logger.info(f"Response to text: {response_it}")
                    score_it = extract_json_from_response(response_it)
                    # 2. add response to the result and feedbacks
                    result[generated_promt][image_path] = {}
                    feedbacks[generated_promt] = []
                    result[generated_promt][image_path]['it'] = score_it
                    overall_score += (score_it['overall_score'] - min((len(score_it['hallucinated_elements']) * self.penalty_score), 1))
                    # 2.1 add the hallucinated elements to the feedbacks
                    feedbacks[generated_promt] += score_it['hallucinated_elements']
                if audio is not None:
                    # 3. judge the image with respect to the audio
                    prompt = f"Judge the following generated image with respect to the audio.\n Generated Prompt: {generated_promt}."
                    message = [
                        {"role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image": os.path.join(base_path, image_path)},
                            {"type": "audio", "audio": audio},
                        ]}
                    ]
                    message = Message.system_message(VERIFICATION_IMAGE_PROMPT) + message
                    response_ai = await self.omniclient.ask(messages=message)
                    logger.info(f"Response to audio: {response_ai}")
                    # 4. add response to the result and feedbacks
                    score_ai = extract_json_from_response(response_ai)
                    result[generated_promt][image_path]['ai'] = score_ai
                    overall_score += (score_ai['overall_score'] - min((len(score_it['hallucinated_elements']) * self.penalty_score), 1))
                    # 4.1 add the hallucinated elements to the feedbacks
                    feedbacks[generated_promt] += score_ai['hallucinated_elements']
                # remove the duplicates
                feedbacks[generated_promt] = list(set(feedbacks[generated_promt]))
                
                if text is not None and audio is not None:
                    overall_score /= 2
                    
                result[generated_promt][image_path]['overall_score'] = overall_score
                
                if overall_score > best_score:
                    # update the best score and text
                    best_score = overall_score
                    best_prompt = generated_promt
                    best_image = image_path
                    result['best_prompt'] = best_prompt
                    result['best_image'] = best_image
                    result['best_score'] = best_score
        # 5. final, check the best image and prompt
        if len(feedbacks[best_prompt]) > 2 and best_score < self.generation_threshold:
            need_to_regenerate = True
           
        return convert_defaultdict(result), need_to_regenerate, feedbacks  
    
    async def _verify_generation_audio(self, image=None, text=None, generated_audios=None, judge_model=None, base_path=None):
        """
        Verify the generation audio using the LMM.
        """
        result = defaultdict(dict)
        feedbacks = {}
        image = self._remove_file_prefix(image) if image else None
        best_score = 0.
        best_prompt = None
        best_audio = None
        result['best_score'] = best_score
        need_to_regenerate = False
        for item in generated_audios:
            generated_promt, audio_paths = item['prompts'], item['generations']
            for audio_path in audio_paths:
                if text is not None:
                    # 1. judge the audio with respect to the text
                    prompt = f"Judge the following generated audio with respect to the generated prompt and ground-truth text.\n Generated Prompt: {generated_promt}.\n Ground-truth Text: {text}."
                    overall_score = 0.
                    message = [
                        {"role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "audio", "audio": os.path.join(base_path, audio_path)},
                        ]}
                    ]
                    message = Message.system_message(VERIFICATION_AUDIO_PROMPT) + message
                    response_at = await self.omniclient.ask(messages=message)
                    logger.info(f"Response to text: {response_at}")
                    score_at = extract_json_from_response(response_at)
                    # 2. add response to the result and feedbacks
                    result[generated_promt][audio_path] = {}
                    # feedbacks[generated_promt] = []
                    result[generated_promt][audio_path]['at'] = score_at
                    overall_score += (score_at['overall_score'] - min((score_at['hallucinated_assertions'] * self.penalty_score), 1))
                # 2.1 add the hallucinated elements to the feedbacks
                # feedbacks[generated_promt] += score_it['hallucinated_elements'] # Audio does not have hallucinated elements
                # 3. judge the image with respect to the audio
                if image is not None:
                    prompt = f"Judge the following generated audio with respect to the image.\n Generated Prompt: {generated_promt}."
                    message = [
                        {"role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image": image},
                            {"type": "audio", "audio": os.path.join(base_path, audio_path)},
                        ]}
                    ]
                    message = Message.system_message(VERIFICATION_AUDIO_PROMPT) + message
                    response_ia = await self.omniclient.ask(messages=message)
                    logger.info(f"Response to audio: {response_ia}")
                    # 4. add response to the result and feedbacks
                    score_ia = extract_json_from_response(response_ia)
                    result[generated_promt][audio_path]['ia'] = score_ia
                    overall_score += (score_ia['overall_score'] - min((score_at['hallucinated_assertions'] * self.penalty_score), 1))
                if image is not None and text is not None:
                    overall_score /= 2
                # 4.1 add the hallucinated elements to the feedbacks
                # feedbacks[generated_promt] += score_ia['hallucinated_elements']
                # remove the duplicates
                # feedbacks[generated_promt] = list(set(feedbacks[generated_promt]))
                
                result[generated_promt][audio_path]['overall_score'] = overall_score
                
                if overall_score > best_score:
                    # update the best score and text
                    best_score = overall_score
                    best_prompt = generated_promt
                    best_audio = audio_path
                    result['best_prompt'] = best_prompt
                    result['best_audio'] = best_audio
                    result['best_score'] = best_score
           
        return convert_defaultdict(result), need_to_regenerate, feedbacks  
    
    def _load_completion_result(self, work_dir: str, target) -> Dict:
        file_path = os.path.join(work_dir, f"completion/{target}/prompt.json")
        if os.path.exists(file_path):
            completion_result = load_json_from_file(file_path)
            return completion_result
        else:
            logger.warning(f"File {file_path} does not exist.")
            raise FileNotFoundError(f"File {file_path} does not exist.")
                
    async def step(self, metadata):
        """
        The metadata contains the following keys:
            metadata = {
                'workflow', 'current_step', 'summary',
                'meida', 'tool', 'target_modality, 'work_dir'
            }
        """
        # unpack metadata
        current_step = metadata['current_step']
        workflow = metadata['workflow']
        step_name = workflow[current_step]
        target_modality = metadata['target_modality']
        work_dir = metadata['work_dir']
        media = metadata['media']
        tool = metadata['tool']
        summary = metadata['summary']
        step = current_step
        feedbacks = None
        base_path = os.path.join(work_dir, f"completion/{target_modality}")
        if 'undersanding' in step_name and not self.only_verify_generation:
            # verify the understanding step
            pass
        elif 'generation' in step_name:
            candidates = self._load_completion_result(work_dir, target_modality)
            if target_modality == 'text':
                # verify the generation text
                generated = [i['text'] for i in candidates['candidates']]
                scores, need_to_regenerate, feedbacks = await self._verify_generation_text(image=media['image'] if 'image' in media else None, 
                                                                                     audio=media['audio'] if 'audio' in media else None, 
                                                                                     generated_texts=generated, 
                                                                                     judge_model=tool)
            elif target_modality == 'image':
                # verify the generation image
                generated = candidates['candidates']
                scores, need_to_regenerate, feedbacks = await self._verify_generation_image(audio=media['audio'] if 'audio' in media else None, 
                                                                                      text=media['text'] if 'text' in media else None, 
                                                                                      generated_images=generated, 
                                                                                      judge_model=tool, 
                                                                                      base_path=base_path)
            elif target_modality == 'audio':
                # verify the generation audio
                generated = candidates['candidates']
                scores, need_to_regenerate, feedbacks = await self._verify_generation_audio(image=media['image'] if 'image' in media else None, 
                                                                                      text=media['text'] if 'text' in media else None, 
                                                                                      generated_audios=generated, 
                                                                                      judge_model=tool, 
                                                                                      base_path=base_path)      
                
            if scores['best_score'] >= self.generation_threshold \
                and not need_to_regenerate:
                step = current_step + 1
                feedbacks = None
                save_json(scores, os.path.join(work_dir, f"completion/{target_modality}/verify.json"))
            else:
                # To prevent the infinite loop, we need to limit the number of generations
                if self.current_generation_step < self.max_generation_verification_step:
                    self.current_generation_step += 1
                    save_json(scores, os.path.join(work_dir, f"completion/{target_modality}/regenerate-{self.current_generation_step}.json"))
                else:
                    step = current_step + 1
                    feedbacks = None
                    save_json(scores, os.path.join(work_dir, f"completion/{target_modality}/verify.json"))
        else:
            step = current_step + 1
        
        if step >= len(workflow):
            return -1, feedbacks
        return step, feedbacks
    
    
    def clear(self):
        """
        Clear the agent state.
        """
        self.current_generation_step = 0
