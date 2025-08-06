"""
OmniUnderstanding Agent
"""
import warnings
warnings.filterwarnings("ignore", module='transformers')

from typing import Any


from afm2.prompts.omni_understanding import SYSTEM_PROMPT, NEXT_STEP_PROMPT
from afm2.schema import Message
from afm2.logger import logger
from afm2.llm_client import LLM, OmniClient




class OmniUnderstanding:
    """
    An agent for understanding images and audio using a language model.
    This agent can process images and generate textual descriptions or analyses.
    """
    
    def __init__(self, llm: LLM, omni_client: OmniClient, knowledges: dict):
        self.llm = llm
        self.omni_client = omni_client
        self.knowledges = knowledges
        self.system_prompt: str = SYSTEM_PROMPT
        self.next_step_prompt: str = NEXT_STEP_PROMPT
        self.pre_check = self.knowledges['pre']
        self.name: str = "omni_understanding"

    
    def _process_input(self, media_path):
        if media_path.startswith("file://"):
            return media_path[len("file://"):]
        else:
            return media_path
        
    def _process_output(self, output):
        return output[len("[ANSWER]:") + 1:].strip()
    
    def format_answers(self, answers: dict):
        title = "The `omni_understanding` agent has provided the following information:"
        body = []
        for q, a in answers.items():
            body.append(f"[QUESTION]: {q}\n[ANSWER]: {a}")
            
        response = f"{title} {''.join(body)}"
        
        return response
    
    async def _for_image(self, knowledge_domain, image=None, audio=None):
        return await self._ask_with_media(knowledge_domain, image=image, audio=audio)
    
    async def _for_audio(self, knowledge_domain, image=None, audio=None):
        answers = {}
        # for prevent to extract low value audio, we should check the audio content first.
        message = [
            {"role": "user",
                "content": [
            {"type": "text", "text": f"[QUESTION]: {self.pre_check[0].format(modality='audio')}"},
            {"type": "audio", "audio": audio}]},
        ]
        system_prompt = self.system_prompt.format(target=knowledge_domain)
        message = Message.system_message(system_prompt) + message
        response = await self.omni_client.ask(messages=message)
        if "no" in response.lower() or "not" in response.lower():
            logger.info(f"The audio has no value.")
            message = [
                {"role": "user",
                "content": [
                    {"type": "text", "text": f"[QUESTION]: What's the audio sound like?"},
                    {"type": "audio", "audio": audio},
                ]}
            ]
            message = Message.system_message(system_prompt) + message
            response = await self.omni_client.ask(messages=message)
            response = self._process_output(response)
            answers[self.pre_check[0].format(modality='audio')] = 'No.'
            answers["What's the audio sound like?"] = response
            summary = "This audio does not contain any meaningful content. " + response
            return answers, summary
        else:
            return await self._ask_with_media(knowledge_domain, image=image, audio=audio)
    
    async def _ask_with_media(self, knowledge_domain, image=None, audio=None):
        answers = {}
        
        for question in self.knowledges[knowledge_domain]:
            message = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"[QUESTION]: {question}"}]
                }
            ]
            if image is not None:
                message[0]['content'].append({"type": "image", "image": image})
            if audio is not None:
                message[0]['content'].append({"type": "audio", "audio": audio})
            
            system_prompt = self.system_prompt.format(target=knowledge_domain)
            message = Message.system_message(system_prompt) + message
            
            response = await self.omni_client.ask(messages=message)
            
            answers[question] = self._process_output(response)
            
        formatted_answers = self.format_answers(answers)
        
        summary_message = Message.user_message(self.next_step_prompt.format(qa_pairs=formatted_answers,
                                                                            target=knowledge_domain))
        
        response = await self.llm.ask(messages=[summary_message])
        summary = self._process_output(response)
        
        logger.info(f"{self.name}'s summary: {summary}")
        
        return answers, summary
    
    async def step(self, image=None, audio=None, knowledge_domain='image') -> Any:
        if image is not None:
            image = self._process_input(image)
        if audio is not None:
            audio = self._process_input(audio)
            
        
        func = self._for_image if knowledge_domain == 'image' else self._for_audio
        answers, summary = await func(knowledge_domain, image=image, audio=audio)
                
        return answers, summary
            