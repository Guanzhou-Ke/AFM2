import json
import os
import numpy as np
import random


def simulate_missing_modality(index, 
                              missing_type, 
                              missing_ratio, 
                              seed=2025,
                              save_path=None):
    """
    Simulate multimodal missing data
    :param data_json: data (json format, including text, audio, image information)
    :param missing_type: missing type ("text", "image", "audio")
    :param missing_ratio: missing ratio (decimal between 0-1)
    :return: index list of missing data
    """
    
    n_samples = len(index)
    missing_count = int(n_samples * missing_ratio)
    
    missing_config = {
        "type": missing_type,
        "missing_ratio": missing_ratio,
    }
    
    # two-element, 1: avaliable, 0: missing.
    # 1st: text, 2nd: image, 3rd: audio.
    missing_details = {}

    random.seed(seed)
    np.random.seed(seed)

    # Select sample indexes for missing text
    missing_indices = random.sample(range(n_samples), missing_count)
    missing_config['num_missing'] = len(missing_indices)
    missing_config['total'] = n_samples
    # Generate missing details.
    if missing_type == "text":
        for idx in missing_indices:
            missing_details[index[idx]] = (0, 1, 1)  # missing text part.
    elif missing_type == "image":
        for idx in missing_indices:
            missing_details[index[idx]] = (1, 0, 1)   # missing Image part.
    elif missing_type == "audio":
        for idx in missing_indices:
            missing_details[index[idx]] = (1, 1, 0)   # missing Audio part.
    else:
        raise ValueError("Invalid missing_type. Choose from 'text', 'image', or 'audio'.")
    
    missing_config["missing_details"] = missing_details
    
    if save_path is not None:
        json.dump(missing_config, open(save_path, 'w'), indent=4)
    
    return missing_config


if __name__ == '__main__':
    import pandas as pd
    import json
    import os

    data = json.load(open('afm2/configs/eval/coco_val.json', 'r'))
    # index = data['audiocap_id'].tolist()
    index = list(range(len(data)))
    
    seed = 2025
    # save_path = 'afm2/configs/missing-info/vggsound'
    # save_path = 'afm2/configs/missing-info/msrvtt'
    # save_path = 'afm2/configs/missing-info/audiocap'
    save_path = 'afm2/configs/missing-info/coco2014'
    
    os.makedirs(save_path, exist_ok=True)
    # for missing_type in ['text', 'image', 'audio']:
    for missing_type in ['text', 'image']:
        for ratio in [0.3, 0.5, 0.7]:
            simulate_missing_modality(index, 
                                      missing_type, 
                                      ratio, 
                                      seed=seed,
                                      save_path=os.path.join(save_path, f'{missing_type}-{ratio}.json'))
        # different ratio shared same seed.
        seed += 1