from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from utils import load_config

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
SRC_DIR = PROJECT_ROOT / 'afm2'  

# LLM config

BASE_URL_CLAUDE = '[Your Claude API URL]'
BASE_URL_OPENAI = '[Your OpenAI API URL]'
BASE_URL_LOCAL = 'http://0.0.0.0:12345/v1'
API_KEY = '[Your API Key]'
API_KEY_LOCAL = 'token-abc123'


# Model List
OPENAI_O3_MINI = 'o3-mini-2025-01-31'
CLAUDE_3_7 = 'claude-3-7-sonnet-thinking'
DEEPSEEK_R1 = 'deepseek-reasoner'
QWQ_32b = 'Qwen/QwQ-32B'



class LLMSettings(BaseModel):
    model: str = Field(..., description="Model name")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")
    max_input_tokens: Optional[int] = Field(
        None,
        description="Maximum input tokens to use across all requests (None for unlimited)",
    )
    temperature: float = Field(1.0, description="Sampling temperature")
    api_type: str = Field(..., description="Openai, or Local")



if __name__ == "__main__":
    config = load_config(SRC_DIR / 'configs' / 'lmm-config.yaml')
    print(LLMSettings.model_construct(**dict(config.local_llm)))