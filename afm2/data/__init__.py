from .datasets import VGGSoundDataset, MSRVTTDataset, AudiocapsDataset, Coco2014Dataset


def load_dataset(dataset_name: str, config):
    """
    Load the specified dataset.
    
    Args:
        dataset_name (str): The name of the dataset to load.
        
    Returns:
        Dataset: The loaded dataset.
    """
    
    if dataset_name == "vggsound":
        
        return VGGSoundDataset(vision_root=config.vision_root, 
                               audio_root=config.audio_root,
                               csv_index=config.csv_index,
                               missing_index=config.missing_index,
                               use_long_caption=config.use_long_caption,)
    elif dataset_name == "msrvtt":
        return MSRVTTDataset(vision_root=config.vision_root,
                             audio_root=config.audio_root,
                             metadata_path=config.metadata_path,
                             missing_index=config.missing_index,
                             label_index=config.label_index,)
    elif dataset_name == 'audiocaps':
        return AudiocapsDataset(vision_root=config.vision_root,
                               audio_root=config.audio_root,
                               csv_index=config.csv_index,
                               missing_index=config.missing_index
                               )
    elif dataset_name == 'coco2014':
        return Coco2014Dataset(vision_root=config.vision_root,
                               class_name_path=config.class_name_path,
                               metadata_path=config.metadata_path,
                               missing_index=config.missing_index
                               )
    elif dataset_name == 'audiocaps-classification':
        train_set = AudiocapsDataset(vision_root=config.train.vision_root,
                               audio_root=config.train.audio_root,
                               csv_index=config.train.csv_index,
                               category_info=config.train.category_info,
                               missing_index=config.train.missing_index
                               )
        test_set = AudiocapsDataset(vision_root=config.test.vision_root,
                               audio_root=config.test.audio_root,
                               csv_index=config.test.csv_index,
                               category_info=config.test.category_info,
                               missing_index=config.test.missing_index
                               )
        return train_set, test_set
    elif dataset_name == 'msrvtt-classification':
        train_set = MSRVTTDataset(vision_root=config.train.vision_root,
                             audio_root=config.train.audio_root,
                             metadata_path=config.train.metadata_path,
                             missing_index=config.train.missing_index,
                             label_index=config.train.label_index,)
        test_set = MSRVTTDataset(vision_root=config.test.vision_root,
                             audio_root=config.test.audio_root,
                             metadata_path=config.test.metadata_path,
                             missing_index=config.test.missing_index,
                             label_index=config.test.label_index,)
        return train_set, test_set
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")