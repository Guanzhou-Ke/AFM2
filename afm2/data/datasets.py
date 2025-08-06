import json
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset



class VGGSoundDataset(Dataset):
    
    def __init__(self, 
                 vision_root, 
                 audio_root, 
                 csv_index, 
                 missing_index=None, 
                 use_long_caption=False) -> None:
        super().__init__()
        self.vision_root = Path(vision_root)
        self.audio_root = Path(audio_root)
        self.csv_index = csv_index
        self.missing_index = missing_index
        
        self.metadata = pd.read_csv(csv_index)
        self.missing_index = json.load(open(missing_index, 'r')) if missing_index else None
        
        self.index = list(self.missing_index['missing_details'].keys())
        self.use_long_caption = use_long_caption
        
        
    def __len__(self):
        return len(self.index)
    
    
    def __getitem__(self, idx):
        clip_id = self.index[idx]
        vision_path = self.vision_root / clip_id / f'{clip_id}_0.jpeg'
        short_caption = self.metadata.loc[self.metadata['youtube_id'] == clip_id, 'short_caption'].values[0]
        long_caption = self.metadata.loc[self.metadata['youtube_id'] == clip_id, 'long_caption'].values[0]
        audio_path = self.audio_root / f'{clip_id}.wav'
        
        missing_list = self.missing_index['missing_details'][clip_id]
        text = long_caption if self.use_long_caption else short_caption
        # 1st: text, 2nd: image, 3rd: audio.
        data = {
            'id': clip_id,
            'vision': str(vision_path) if missing_list[1] == 1 else None,
            'audio': str(audio_path) if missing_list[2] == 1 else None,
            'text': text if missing_list[0] == 1 else None,
            'gt_vision': str(vision_path),
            'gt_audio': str(audio_path),
            'gt_text': text,
        }
        
        return data
    
    
class MSRVTTDataset(Dataset):
    
    def __init__(self, 
                 vision_root, 
                 audio_root, 
                 metadata_path, 
                 missing_index=None, 
                 label_index=None,
                 use_long_caption=False) -> None:
        super().__init__()
        self.vision_root = Path(vision_root)
        self.audio_root = Path(audio_root)
        self.missing_index = missing_index
        
        self.metadata = json.load(open(metadata_path, 'r'))
        self.missing_index = json.load(open(missing_index, 'r')) if missing_index else None
        
        self.index = list(self.missing_index['missing_details'].keys()) if missing_index else list(range(len(self.metadata)))
        self.use_long_caption = use_long_caption
        
        self.label_info = json.load(open(label_index, 'r'))['label_name'] if label_index else None
        self.class_nums = len(self.label_info) if self.label_info else None
        
        
        
    def __len__(self):
        return len(self.index)
    
    
    def __getitem__(self, idx):
        clip_item = self.metadata[int(self.index[idx])]
        clip_id = clip_item['video_id']
        vision_path = self.vision_root / clip_id / f'{clip_id}_0.jpeg'
        text = clip_item['caption'][0]
        audio_path = self.audio_root / f'{clip_id}.wav'
        label = clip_item['label']
        label_idx = self.label_info.index(label) if self.label_info else None
        if self.missing_index is not None:
            missing_list = self.missing_index['missing_details'][self.index[idx]]
        else:
            missing_list = [1, 1, 1]
        # 1st: text, 2nd: image, 3rd: audio.
        data = {
            'id': clip_id,
            'vision': str(vision_path) if missing_list[1] == 1 else None,
            'audio': str(audio_path) if missing_list[2] == 1 else None,
            'text': text if missing_list[0] == 1 else None,
            'gt_vision': str(vision_path),
            'gt_audio': str(audio_path),
            'gt_text': text,
            'label': label,
            'label_idx': label_idx
        }
        
        return data


class AudiocapsDataset(Dataset):
    
    def __init__(self, 
                 vision_root, 
                 audio_root, 
                 csv_index, 
                 category_info=None,
                 missing_index=None) -> None:
        super().__init__()
        self.vision_root = Path(vision_root)
        self.audio_root = Path(audio_root)
        self.csv_index = csv_index
        self.missing_index = missing_index
        
        if category_info is not None:
            self.category_info = json.load(open(category_info, 'r'))
            self.class_nums = len(self.category_info['label_names'])
        else:
            self.category_info = None
            
        
        
        self.metadata = pd.read_csv(csv_index)
        self.missing_index = json.load(open(missing_index, 'r')) if missing_index else None
        
        self.index = list(self.missing_index['missing_details'].keys()) if missing_index else list(self.metadata['audiocap_id'].values)
        
        
    def __len__(self):
        return len(self.index)
    
    
    def __getitem__(self, idx):
        clip_id = self.index[idx]
        vision_path = self.vision_root / str(clip_id) / f'{clip_id}_0.jpeg'
        text = self.metadata.loc[self.metadata['audiocap_id'] == int(clip_id), 'caption'].values[0]
        audio_path = self.audio_root / f'{clip_id}.wav'
        label = self.metadata.loc[self.metadata['audiocap_id'] == int(clip_id), 'label'].values[0]
        label_idx = self.metadata.loc[self.metadata['audiocap_id'] == int(clip_id), 'label_idx'].values[0]
        label_idx = [int(_) for _ in label_idx.split(',')]
        if self.missing_index is not None:
            missing_list = self.missing_index['missing_details'][clip_id]
        else:
            missing_list = [1, 1, 1]
        # 1st: text, 2nd: image, 3rd: audio.
        data = {
            'id': clip_id,
            'vision': str(vision_path) if missing_list[1] == 1 else None,
            'audio': str(audio_path) if missing_list[2] == 1 else None,
            'text': text if missing_list[0] == 1 else None,
            'gt_vision': str(vision_path),
            'gt_audio': str(audio_path),
            'gt_text': text,
            'label': label,
            'label_idx': label_idx
        }
        
        return data
    
    
class Coco2014Dataset(Dataset):
    
    def __init__(self, 
                 vision_root,
                 class_name_path: Path, 
                 metadata_path: Path, 
                 missing_index=None) -> None:
        super().__init__()
        self.vision_root = Path(vision_root)
        self.class_name = json.load(open(class_name_path, 'r'))
        self.metadata = json.load(open(metadata_path, 'r'))
        self.missing_index = missing_index
        
        self.missing_index = json.load(open(missing_index, 'r')) if missing_index else None
        
        self.index = list(self.missing_index['missing_details'].keys()) if missing_index else list(range(len(self.metadata)))
        
        
    def __len__(self):
        return len(self.index)
    
    
    def __getitem__(self, idx):
        clip_id = int(self.index[idx])
        vision_path = self.vision_root / self.metadata[clip_id]['file_name']
        text = self.metadata[clip_id]['captions'][0]
        label_idx = self.metadata[clip_id]['labels']
        if self.missing_index is not None:
            missing_list = self.missing_index['missing_details'][str(clip_id)]
        else:
            missing_list = [1, 1, 1]
        # 1st: text, 2nd: image
        data = {
            'id': clip_id,
            'vision': str(vision_path) if missing_list[1] == 1 else None,
            'text': text if missing_list[0] == 1 else None,
            'audio': None,
            'gt_audio': None,
            'gt_vision': str(vision_path),
            'gt_text': text,
            'label': [self.class_name['label_name'][_] for _ in label_idx],
            'label_idx': label_idx
        }
        
        return data
    

if __name__ == '__main__':
    
    
    vision_root = '[Path to Your Data]/MSR-VTT/images'
    audio_root = '[Path to Your Data]/MSR-VTT/videos/all'
    metadata_path = 'afm2/configs/eval/msrvtt_test_metadata.json'
    missing_index = 'afm2/configs/missing-info/msrvtt/audio-0.3.json'
    label_index = 'afm2/configs/eval/msrvtt_category.json'
    dataset = MSRVTTDataset(vision_root, audio_root, metadata_path, missing_index, label_index=label_index)
    print(len(dataset))
    print(dataset[150])
    
    
    pass
    
  
    
    
    