import argparse
import os
from pathlib import Path
from time import time
import asyncio
from tqdm import tqdm

from afm2.agents.mainagent import MainAgent
from afm2.data import load_dataset
from afm2.utils import load_config, save_json, load_json_from_file
from afm2.constants import SRC_DIR
from afm2.metrics import metrics_audio, metrics_language, metrics_vision


async def create_agent():
    return await MainAgent().initialize_agent()



async def process_model_step(agent, messages, work_dir, gt_item):
    """Run the model step with the given messages."""
    return await agent.step(messages, work_dir=work_dir, gt_item=gt_item)



def _construct_messages(image_path=None, audio_path=None, text=None, complete_target='image'):
    """Message should be a list of dictionaries.
        [{'type': 'image', 'image': 'file:///tmp/xxx/xxx.jpeg'}, 
        {'type': 'audio', 'audio': 'file:///tmp/xxx/xxx.wav'}, 
        {'type': 'ava_text', 'ava_text': 'xxxx'},
        {'type': 'text', 'text': 'complete the missing'}]"""
    message = []
    if image_path is not None:
        if image_path.startswith('file://'):
            message.append({'type': 'image', 'image': image_path})
        else:
            message.append({'type': 'image', 'image': 'file://' + image_path})
    if audio_path is not None:
        if audio_path.startswith('file://'):
            message.append({'type': 'audio', 'audio': audio_path})
        else:
            message.append({'type': 'audio', 'audio': 'file://' + audio_path})
    if text is not None:
        message.append({'type': 'ava_text', 'ava_text': text})
    
    prompt = f'Complete the missing {complete_target}.'
    message.append({'type': 'text', 'text': prompt})
    return [{'role': 'user', 'content': message}]


def parase_args():
    parser = argparse.ArgumentParser(description="Load and run the model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="vggsound",
        help="Evaluation dataset.",
    )
    return parser.parse_args()


async def eval(dataset, base_dir, complete_target='image', retry=3):
    # data_len = len(dataset)
    data_len = 2
    start_time = time()
    preds = []
    targets = []
    # 1. create agents.
    agents = []
    tasks = [create_agent() for _ in range(data_len)]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Creating agents"):
        agent = await coro
        agents.append(agent)
    
    messages = []
    work_dirs = []
    gt_items = []
    # 2. construct the messages.
    for i in tqdm(range(data_len)):
        # To support breakpoint.
        item = dataset[i]
        clip_id = str(item['id'])
        work_dir = base_dir / clip_id
        
        if not os.path.exists(work_dir / 'gt.json'):
            image_path = item['vision']
            audio_path = item['audio']
            text = item['text']
            # 1. Construct the messages.
            msg = _construct_messages(image_path=image_path, audio_path=audio_path, text=text, complete_target=complete_target)
            messages.append(msg)
            work_dirs.append(work_dir)
            gt_items.append(item)
        # print(f"Constructed messages for {clip_id} on {work_dir} ({i+1}/{data_len})")
    
    # 3. wrap coroutines for each agent step.
    tasks = [process_model_step(agent, msg, work_dir, gt_item) for agent, msg, work_dir, gt_item in zip(agents, messages, work_dirs, gt_items)]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing samples"):
        result = await coro
        item = result['gt_item']
        verify = load_json_from_file(result['verifier_json'])
        if complete_target == 'image':
            preds.append(str(work_dir / 'completion' / complete_target / verify['best_image']))
            targets.append(str(item['gt_vision']))
        elif complete_target == 'audio':
            preds.append(str(work_dir / 'completion' / complete_target / verify['best_audio']))
            targets.append(str(item['gt_audio']))
        elif complete_target == 'text':
            preds.append(str(work_dir / 'completion' / complete_target / verify['best_text']))
            targets.append(str(item['gt_text']))
        else:
            raise ValueError(f"Unknown complete target: {complete_target}")
        
    # print("All agents have completed their steps.")
    # # 4. Save the results.
    # print(results)
    # load the generated results
    
        
    save_json({
        'preds': preds,
        'targets': targets,
    }, os.path.join(base_dir, 'preds.json'))
    
    # 4. Calculate the metrics.
    if complete_target == 'audio':
        metrics = metrics_audio(preds, targets)
    elif complete_target == 'image':
        metrics = metrics_vision(preds, targets)
    elif complete_target == 'text':
        metrics = metrics_language(preds, targets)
    else:
        raise ValueError(f"Unknown complete target: {complete_target}")
    
    total_time = time() - start_time
    metrics['total_time_in_h'] = total_time / 3600
    metrics['avg_time_in_s'] = total_time / data_len
    print("Metrics:", metrics)
    save_json(metrics, os.path.join(base_dir, 'evaluation.json'), )
    return metrics

    
        
        


    
if __name__ == "__main__":
    args = parase_args()
    # 1. Initialize the model and load the dataset.
    config = load_config(SRC_DIR / 'configs' / 'eval' / f"{args.dataset}.yaml")
    dataset = load_dataset(args.dataset, config.dataset)
    
    # 2. Create the work directory.
    work_dir = Path(config.default.work_dir) / config.default.exp_name
    os.makedirs(work_dir, exist_ok=True)
    
    # print(f"Work directory: {work_dir}")
    
    # 3. Iterate through the dataset and process each item.
    asyncio.run(eval(dataset, work_dir, complete_target=config.default.complete_target, retry=config.default.retry))