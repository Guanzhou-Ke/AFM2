from torchvision import transforms
from transformers import Qwen2VLImageProcessor
from qwen_vl_utils import process_vision_info


def get_normal_image_transform(image_size=224):
    """
    Get the image transform for the image data.
    Args:
        image_size: the size of the image, default is 224.
    Returns:
        transform: the transform for the image data.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    
def get_image_preprocessor(name='Qwen/Qwen2.5-VL-7B-Instruct'):
    if name.startswith('Qwen'):
        processor = Qwen2VLImageProcessor.from_pretrained(name)
    else:
        raise ValueError("The image processor name is not supported.")
    return processor
    
    
if __name__ == "__main__":
    # test the image transform.
    image_transform = get_normal_image_transform()
    print(image_transform)
    
    # test the image processor.
    processor = get_image_preprocessor()
    print(processor)