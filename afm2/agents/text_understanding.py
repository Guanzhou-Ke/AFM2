"""
Impletement all tools here.
"""
from typing import Any


from afm2.prompts.text_understanding import SYSTEM_PROMPT, NEXT_STEP_PROMPT
from afm2.schema import Message
from afm2.logger import logger
from afm2.llm_client import LLM





class TextUnderstanding:
    """
    An agent for understanding text using a language model.
    This agent can process text and generate textual descriptions or analyses.
    """
    
    

    def __init__(self, llm: LLM, knowledges):
        self.llm = llm
        self.knowledges = knowledges
        self.system_prompt: str = SYSTEM_PROMPT
        self.next_step_prompt: str = NEXT_STEP_PROMPT
        self.name: str = "text_understanding"
    
    
        
    def _process_output(self, output):
        return output[len("[ANSWER]:") + 1:].strip()
    
    def format_answers(self, answers: dict):
        title = "The `image_understanding` agent has provided the following information:"
        body = []
        for q, a in answers.items():
            body.append(f"[QUESTION]: {q}\n[ANSWER]: {a}")
            
        response = f"{title} {''.join(body)}"
        
        return response
    
    async def step(self, text) -> Any:
        answers = {}
        
        for question in self.knowledges:
            message = [
                {"role": "user",
                "content": [
                    {"type": "text", "text": f"[USER PROVIDING TEXT]: {text}\n[QUESTION]: {question}"},
                ]}
            ]
         
            message = Message.system_message(self.system_prompt) + message
            
            response = await self.llm.ask(messages=message)
            
            answers[question] = self._process_output(response)
            
        formatted_answers = self.format_answers(answers)
        
        summary_message = Message.user_message(self.next_step_prompt.format(qa_pairs=formatted_answers))
        
        response = await self.llm.ask(messages=[summary_message])
        
        summary = self._process_output(response)
        
        logger.info(f"{self.name}'s summary: {summary}")
                
        return answers, summary
            