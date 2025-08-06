import json
import re

from omegaconf import OmegaConf
import torch
import numpy as np
import httpx
from typing import Dict, Optional


async def generate_tool_calling(
    server_url: str,
    prompt: str,
    output_path: str,
    save_name: str,
    timeout: int = 600,
    extra_payload: Optional[Dict] = None,
) -> Dict:
    
    payload = {
        "prompt": prompt,
        "output_path": str(output_path),
        "save_name": save_name,
    }

    if extra_payload:
        payload.update(extra_payload)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(server_url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {"error": str(e)}


def convert_result_to_json(result: str):
    result = result.replace('```json', '').replace('```', '')
    return json.loads(result)


def load_json_from_file(path: str):
    return json.load(open(path, 'r'))

def save_json(data, path: str):
    return json.dump(data, open(path, 'w'), indent=4)


def load_config(path):
    config = OmegaConf.load(path)
    return config


def reproducibility_setting(seed):
    """
    set the random seed to make sure reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)

    
    
def count_parameters(model: torch.nn.Module, dtype):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_per_param = 2 if dtype == torch.float16 else 4  # 2 字节用于 FP16，4 字节用于 FP32
    total_memory_MB = total_params * bytes_per_param / (1024 ** 2)
    return total_params, total_memory_MB


def extract_json_from_response(response: str) -> dict:
    """
    Extracts the first JSON object found in `response` by manual brace-matching
    and parses it.

    Args:
        response: The raw string returned by your verifier agent.

    Returns:
        A Python dict containing the parsed JSON.

    Raises:
        ValueError: If no valid JSON object can be found or parsed.
    """
    # Find the first opening brace
    start = response.find('{')
    if start == -1:
        raise ValueError("No JSON object found in response.")

    depth = 0
    in_string = False
    escape = False
    end = None

    # Scan character by character
    for i, c in enumerate(response[start:], start):
        if escape:
            escape = False
        elif c == '\\':
            escape = True
        elif c == '"' and not escape:
            in_string = not in_string
        elif not in_string:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

    if end is None:
        raise ValueError("Could not find matching closing '}' for JSON object.")

    json_str = response[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Extracted text is not valid JSON: {e}")


def extract_refined_prompt(llm_response: str) -> str:
    """
    Extracts the text following 'Refined Prompt:' from a multimodal LLM response.

    Parameters:
        llm_response (str): The full response text returned by the LLM.

    Returns:
        str: The refined prompt, or an empty string if not found.
    """
    # Look for 'Refined Prompt:' and capture everything that follows
    pattern = r"Refined Prompt:\s*(.+)$"
    match = re.search(pattern, llm_response, re.DOTALL | re.MULTILINE)
    if not match:
        return ""
    # Strip any leading/trailing whitespace
    return match.group(1).strip()