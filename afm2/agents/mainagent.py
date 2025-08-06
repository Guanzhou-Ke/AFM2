import os
from typing import Dict
from time import time
from pydantic import Field, model_validator

from afm2.agents.base import BaseAgent
from afm2.prompts.mainagent import SYSTEM_PROMPT, NEXT_STEP_PROMPT, SYSTEM_PROMPT_WITHOUT_GRAPH
from afm2.schema import Message, Memory
from afm2.logger import logger
from afm2.llm_client import LLM, OmniClient
from afm2.agents.text_understanding import TextUnderstanding
from afm2.agents.onmi_understanding import OmniUnderstanding
from afm2.agents.text_generation import TextGeneration
from afm2.agents.image_generation import ImageGeneration
from afm2.agents.audio_generation import AudioGeneration
from afm2.agents.verifier import Verifier
from afm2.utils import load_config, load_json_from_file, save_json, extract_json_from_response
from afm2.constants import SRC_DIR


async def async_retry(func, max_retries=3, *args, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"[Retry {attempt}/{max_retries}] Function `{func.__name__}` failed: {e}. Retrying...")
            else:
                logger.error(f"[Retry {attempt}/{max_retries}] Function `{func.__name__}` failed permanently.")
                raise e


class MainAgent(BaseAgent):
    """
    A versatile multimodal agent that uses planning and tools to complete the missing modality.

    
    """

    name: str = "MainAgent"
    description: str = (
        "A versatile multimodal agent that uses planning and tools to complete the missing modality."
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20


    # Add general-purpose tools to the tool collection
    TOOLS: Dict = Field(default={
        "text_understanding": None,
        "image_understanding": None,
        "audio_understanding": None,
        "text_generation": None,
        "image_generation": None,
        "audio_generation": None,
    }, description="List of available tools")
    
    
    @model_validator(mode="after")
    async def initialize_agent(self) -> "MainAgent":
        """Initialize agent with default settings if not provided."""
        if not self.initialized:
            self.config = load_config(SRC_DIR / 'configs' / f'{self.name.lower()}.yaml')
            self.knowledges = load_json_from_file(SRC_DIR / 'knowledges' / self.config.default.knowledge)
            self.use_rag = self.config.default.use_rag
            self.state_history = []
            self.workflow = []
            if self.use_rag:
                self.system_prompt = SYSTEM_PROMPT
            else:
                self.system_prompt = SYSTEM_PROMPT_WITHOUT_GRAPH

            if self.llm is None or not isinstance(self.llm, LLM):
                self.llm = LLM(config_name=self.name.lower())
            if self.omniclient is None:
                self.omniclient = OmniClient(base_url=self.config.omni.base_url,
                                            api_key=self.config.omni.api_key,
                                            model_name=self.config.omni.model,
                                            max_tokens=self.config.omni.max_tokens,)
            if not isinstance(self.memory, Memory):
                self.memory = Memory()
                
            logger.info("Initializing image generation tool...")
            self.TOOLS['image_generation'] = ImageGeneration(self.llm, self.config.generator.image_generator)
            logger.info("Image generation tool initialized.")
                
            logger.info("Initializing audio generation tool...")
            self.TOOLS['audio_generation'] = AudioGeneration(self.llm, self.config.generator.audio_generator)
            logger.info("Audio generation tool initialized.")    
            
            if self.config.default.use_omni_model:
                logger.info("Initializing omni understanding tool...")
                omni_agent = OmniUnderstanding(llm=self.llm,
                                              omni_client=self.omniclient,
                                              knowledges=self.knowledges)
                self.TOOLS['image_understanding'] = omni_agent
                self.TOOLS['audio_understanding'] = omni_agent
                logger.info("Omni understanding tool initialized.")
                self.use_omni_model = True
  
            
            logger.info("Initializing text understanding tool...")
            self.TOOLS['text_understanding'] = TextUnderstanding(llm=self.llm, knowledges=self.knowledges['text'])
            logger.info("Text understanding tool initialized.")
            
            
            logger.info("Initializing text generation tool...")
            self.TOOLS['text_generation'] = TextGeneration(llm=self.llm)
            logger.info("Text generation tool initialized.")
            
            
            
            logger.info("Initializing verifier tool...")
            self.verifier = Verifier(omniclient=self.omniclient, 
                                     penalty_score=self.config.verifier.penalty_score,
                                     generation_threshold=self.config.verifier.generation_threshold,
                                     understanding_threshold=self.config.verifier.understanding_threshold,
                                     max_generation_verification_step=self.config.verifier.max_generation_verification_step,
                                     only_verify_generation=self.config.verifier.only_verify_generation,)
            logger.info("Verifier tool initialized.")
            
            self.initialized = True
        return self
    
    def _clear_all(self):
        self.state_history = []
        self.workflow = []
        self.current_step = 0
        self.ava_modalities = {}
        self.target_modality = None
        self.memory.clear()
        self.verifier.clear()
    
    def _process_input(self, message):
        """extract modalities from the input message."""
        ava_message = []
        modalitis = {}
        for msg in message:
            if msg["role"] == "user":
                for c in msg["content"]:
                    if c['type'] in ['image', 'audio', 'ava_text']:
                        modalitis[c['type'] if c['type'] != 'ava_text' else 'text'] = c[c['type']] 
                    else:
                        ava_message.append({'role': 'user', 'content': c['text']})
            else:
                ava_message.append(msg)
        return ava_message, modalitis
    
    def _extract_workflow(self, response: str):
        """Extract the workflow from the response."""
        # steps_str = response.split('```')[1].strip()[4:].replace('\n', '')
        # steps = json.loads(steps_str)
        steps = extract_json_from_response(response)
        workflow = []
        for i in range(len(steps)):
            if steps[f'step {i+1}'] != 'verifier':
                workflow.append(steps[f'step {i+1}'])
        return workflow
    
    
    def _verify_workflow(self, ava_modal, workflow):
        """Verify the workflow can address the user's needs."""
        # Check if the workflow contains all necessary steps
        required_steps = [f'{s}_understanding' for s in ava_modal]
        required_steps.append(f'{self.target_modality}_generation')
        for step in required_steps:
            if step not in workflow:
                return False
        return True
        
    
    async def planning(self, message):
        # Call the LLM with the system prompt and user message
        if isinstance(message, list):
            ava_message, modalitis = self._process_input(message)
            self.ava_modalities = modalitis
            assert len(modalitis) <= 2, "Too many modalities provided by the user."
            if len(modalitis) == 1:
                ava_modal_type = list(modalitis.keys())[0]
                full_modal = {'image', 'text'}
            elif len(modalitis) == 2:
                ava_modal_type = ' and '.join(modalitis.keys())
                full_modal = {'image', 'text', 'audio'}
            self.target_modality = list(full_modal - set(modalitis.keys()))[0]
            if 'text' in ava_modal_type:
                ava_modal_type += f":({modalitis['text']})"
                
                
            last_msg = ava_message[-1]
            ava_message.pop()
            
            next_step = self.next_step_prompt.format(
                modalities=ava_modal_type,
                user_message=last_msg['content']
            )

            message = Message.system_message(self.system_prompt) + ava_message + Message.user_message(next_step)
            logger.info("Memeory updated with initial message.") 
            self.memory.add_message(message)
        else:
            raise ValueError("Message should be a list of dictionaries.")
        
        workflow = {}
        ava_modal = list(self.ava_modalities.keys())
        attempt = 0
        while not self._verify_workflow(ava_modal, workflow):
            if attempt == 1:
                message.append(
                    Message.user_message("The workflow is not valid. Please refine the workflow.")
                )
            response = await self.llm.ask(
                messages=message
            )
            workflow = self._extract_workflow(response)
            attempt += 1
            
                
        # response = self.llm.ask(
        #     messages=message
        # )
        
        logger.info("Memeory updated with assistant planning.") 
        self.memory.add_message(Message.assistant_message(response))
        
        # self.workflow = self._extract_workflow(response)
        self.workflow = workflow
        self.current_step = 0
        
        logger.info(f"Extracted workflow: {self.workflow}")
        
        return response
    
    async def step(self, messages, work_dir='./test_dir', gt_item=None):
        """Process current state and decide next actions with appropriate context.
        
        Message should be a list of dictionaries.
        [{'type': 'image', 'image': 'file:///tmp/xxx/xxx.jpeg'}, 
        {'type': 'audio', 'audio': 'file:///tmp/xxx/xxx.wav'}, 
        {'type': 'ava_text', 'ava_text': 'xxxx'},
        {'type': 'text', 'text': 'complete the missing'}]
        """
        # 1. do planning.
        await self.planning(messages)
        
        # 2. excute the workflow.
        step_num = len(self.workflow)
        
        summaries = {}
        qa_pairs = {}
        verifier_feedbacks = None
        while True:
            if self.current_step == -1:
                logger.info("All steps completed.")
                break
            tool = self.workflow[self.current_step]
            logger.info(f"Current step: {tool}")
            self.state_history.append(tool)
            if tool not in self.TOOLS:
                raise ValueError(f"Tool {tool} is not available.")
            if 'understanding' in tool:
                if self.use_omni_model and 'text' not in tool:
                    image = self.ava_modalities['image'] if 'image' in self.ava_modalities else None
                    audio = self.ava_modalities['audio'] if 'audio' in self.ava_modalities else None
                    # qa_pair, summary = await self.TOOLS[tool].step(image, audio, knowledge_domain=tool.split('_')[0])
                    qa_pair, summary = await async_retry(self.TOOLS[tool].step, max_retries=3, image=image, audio=audio, knowledge_domain=tool.split('_')[0])
                else:
                    # qa_pair, summary = await self.TOOLS[tool].step(self.ava_modalities[tool.split('_')[0]])
                    
                    qa_pair, summary = await async_retry(self.TOOLS[tool].step, 3, self.ava_modalities[tool.split('_')[0]])
                summaries[tool] = summary
                qa_pairs[tool] = qa_pair
                # end = time()
                # self.memory.add_message(Message.assistant_message(summary))
                # yield f"\n---\nThe tool `{tool}`'s thinking result: " + summary + f"\n--- ({end - start:.2f} seconds.)"
                # logger.info(f"Tool `{tool}` took {end - start:.2f} seconds.")
            # elif 'knowledge_extractor' in tool:   
            #     # qa_pair, summary = await self.TOOLS[tool].step(summaries, 
            #     #                                 self.knowledges[self.target_modality], 
            #     #                                 self.target_modality,
            #     #                                 work_dir)
            #     qa_pair, summary = await async_retry(
            #         self.TOOLS[tool].step, 3, summaries, 
            #         self.knowledges[self.target_modality], 
            #         self.target_modality,
            #         work_dir
            #     )
            #     summaries[tool] = summary
            #     qa_pairs[tool] = qa_pair
            #     # end = time()
            #     self.memory.add_message(Message.assistant_message(summary))
                # yield f"\n---\nThe tool `{tool}`'s thinking result: " + summary + f"\n--- ({end - start:.2f} seconds.)"
                # logger.info(f"Tool `{tool}` took {end - start:.2f} seconds.")       
            elif 'generation' in tool:
                # response = await self.TOOLS[tool].step(summaries, work_dir, self.use_rag, feedbacks=verifier_feedbacks)
                response = await async_retry(
                    self.TOOLS[tool].step, 3, summaries, work_dir, self.use_rag, feedbacks=verifier_feedbacks
                )
                # end = time()
                # yield f"\n---\nThe tool `{tool}`'s thinking result: " + response + f"\n--- ({end - start:.2f} seconds.)"
                # logger.info(f"Tool `{tool}` took {end - start:.2f} seconds.")
                # self.memory.add_message(Message.assistant_message(response))
            
            metadata = {
                'workflow': self.workflow,
                'current_step': self.current_step,
                'summary': summaries,
                'media': self.ava_modalities,
                'tool': self.TOOLS['image_understanding'],
                'target_modality': self.target_modality,
                'work_dir': work_dir
            }
                
            # self.current_step, verifier_feedbacks = await self.verifier.step(metadata)
            self.current_step, verifier_feedbacks = await async_retry(
                self.verifier.step, 3, metadata
            )
            self.state_history.append('verifier')
        save_json(
            {
                'state_history': self.state_history,
                'workflow': self.workflow,
                'QA_pairs': qa_pairs,
                'summaries': summaries,
            },
            os.path.join(work_dir, 'excute_logging.json')
        )
        
        save_json(gt_item, work_dir / 'gt.json')    
        
        return {'state_history': self.state_history, 'workflow': self.workflow, 'QA_pairs': qa_pairs,'summaries': summaries,
                'verifier_json': f'{os.path.join(work_dir, "completion", self.target_modality, "verify.json")}',
                'gt_item': gt_item}

