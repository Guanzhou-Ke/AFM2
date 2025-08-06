import os
import shutil
from typing import List

import torch
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
# from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.text import BLEUScore, ROUGEScore
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import clip
import librosa
from scipy.spatial.distance import cosine
import numpy as np
from torchvision import transforms
import torch_fidelity
from torchmetrics.text import MatchErrorRate




def copy_resize_images(image_paths: List[str], target_dir: str, save_name_index=-4):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    copied_paths = []

    try:
        for path in image_paths:
            if os.path.isfile(path):
                clip_id = path.split('/')[save_name_index]
                try:
                    with Image.open(path) as img:
                        img = img.convert("RGB")  # 确保是RGB格式
                        img = img.resize((256, 256))
                        
                        filename = os.path.basename(path)
                        suffix = filename.split('.')[-1]

                        dst_path = os.path.join(target_dir, f'{clip_id}.{suffix}')
                        img.save(dst_path)
                        copied_paths.append(dst_path)
                except Exception as img_err:
                    raise img_err
            else:
                print(f"File {path} does not exist.")
                continue
    except Exception as e:
        raise e
    return copied_paths


def load_images_uint8(
    paths: list[str],
    resize: bool = True,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Load images from paths.
    """
    pil_transforms = []
    if resize:
        pil_transforms.append(transforms.Resize((299, 299)))
    # PILToTensor: 输出 uint8 的 C×H×W
    pil_transforms.append(transforms.PILToTensor())
    transform = transforms.Compose(pil_transforms)

    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))  # dtype=torch.uint8
    batch = torch.stack(imgs, dim=0)  # (N, C, H, W)
    return batch.to(device) if device else batch


def metrics_is_fid(gen_images, real_images, normalize=False):
    """
    Compute Inception Score and FID between fake and real images.
    Args:
        fake_images: the generated images, shape (N, C, 299, 299).
        real_images: the real images, shape (N, C, 299, 299).
        normalize: indicates the input images are normalized or not.
    Returns:
        result: a dictionary containing the Inception Score (mean and std) and FID
        The keys are "is_mean", "is_std", and "fid".
        IS is the Inception Score, the higher the better, the higher value indicates better quality and diversity.
        FID is the Frechet Inception Distance, the lower the better.
    """
    _, _, H, W = gen_images.shape
    if H != 299 or W != 299:
        raise ValueError("The height and width of the image should be 299.")
    result = {}
    # # IS
    is_score = InceptionScore(feature=2048, normalize=normalize)
    # # is_value = is_score(gen_images)
    # # result["is_mean"] = is_value[0].item()
    # # result["is_std"] = is_value[1].item()
    
    # is_score.update(gen_images)
    # result["is_mean"], result["is_std"] = is_score.compute()
    
    # FID
    fid_score = FrechetInceptionDistance(feature=2048, normalize=normalize)
    # This metric is known to be unstable in its calculatations, 
    # and we recommend for the best results using this metric that you calculate using torch.float64 
    fid_score.set_dtype(torch.float64)
    fid_score.update(real_images, real=True)
    fid_score.update(gen_images, real=False)
    fid_value = fid_score.compute()
    result["fid"] = fid_value.item()
    
    return result




def metric_pesq(preds, target):
    """
    Compute Perceptual Evaluation of Speech Quality (PESQ) between two audio signals.
    Args:
        preds: the predicted audio signal, shape (N,).
        target: the target audio signal, shape (N,).
    Returns:
        result: a dictionary containing the PESQ scores for narrowband and wideband.
        The keys are "pesq_nb" and "pesq_wb".
        The value is the PESQ score, the higher the better. (- 0.5 is the lowest score, 4.5 is the highest score.)
    """
    
    if len(preds) != len(target):
        raise ValueError("The length of the predicted and target audio signals should be the same.")
    
    result= {}
    # sample_rate: the sample rate of the audio signals, default is 8000.
    # mode: the mode of PESQ, 'nb' for narrowband (8 kHz), 'wb' for wideband (16 kHz).
    try:
        nb_value = perceptual_evaluation_speech_quality(preds, target, 8000, "nb")
        # nb_value = nb_pesq(preds, target)
        result["pesq_nb"] = nb_value.item()
    except Exception as e:
        result["pesq_nb"] = 0
    
    # wb_pesq = perceptual_evaluation_speech_quality(16000, "wb")
    # wb_value = wb_pesq(preds, target)
    try:
        wb_value = perceptual_evaluation_speech_quality(preds, target, 16000, "wb")
        result["pesq_wb"] = wb_value.item()
    except Exception as e:
        result["pesq_wb"] = 0
    
    return result


def metrics_language(preds, target, device='cpu'):
    """
    Compute language scores between predicted and target texts.
    Args:
        preds: the predicted texts, shape (N,).
        target: the target texts, shape (N,).
    Returns:
        result: a dictionary containing the BLEU-1 and BLEU-4 scores and ROUGE-L score.
        The keys are "bleu_1", "bleu_4", and "rouge_l".
        The value is the BLEU or ROUGE score, the higher the better.
    """
    result = {}
    bleu_1 = BLEUScore(n_gram=1)
    bleu_4 = BLEUScore(n_gram=4)
    
    rouge_l = ROUGEScore(rouge_keys="rougeL")
    
    clip_score = CLIPScore(download_root=None, device=device, percentage=False)
    
    bleu_1_value = bleu_1(preds, target)
    bleu_4_value = bleu_4(preds, target)
    rouge_l_value = rouge_l(preds, target)
    
    result["bleu_1"] = bleu_1_value.item()
    result["bleu_4"] = bleu_4_value.item()
    result["rouge_l"] = rouge_l_value['rougeL_fmeasure'].item()
    result["rouge_l_precision"] = rouge_l_value['rougeL_precision'].item()
    result["rouge_l_recall"] = rouge_l_value['rougeL_recall'].item()
    
    mer = MatchErrorRate()
    mer_value = mer(preds, target)
    result["mer"] = mer_value.item()
    
    clip_ts = []
    for pred, gt in zip(preds, target):
        clip_t = clip_score.score_same_modality(pred, gt, 'text')
        clip_ts.append(clip_t)
    
    result["clip_t"] = np.mean(clip_ts)
    return result


def load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    return y


def waveform_mse(x, y):
    mse = np.mean((x - y)**2)
    return mse


def mfcc_cosine_similarity(audio1, audio2, sr=16000, n_mfcc=13):

    mfcc_x = librosa.feature.mfcc(y=audio1, sr=sr, n_mfcc=n_mfcc)
    mfcc_y = librosa.feature.mfcc(y=audio2, sr=sr, n_mfcc=n_mfcc)

    avg_x = np.mean(mfcc_x, axis=1)
    avg_y = np.mean(mfcc_y, axis=1)

    similarity = 1 - cosine(avg_x, avg_y)
    return similarity


def metrics_audio(preds, targets, device='cpu'):
    """
    Args:
      pred: prediction audio path.
      target: target audio path
    
    """

    if len(preds) != len(targets):
        raise ValueError("The length of the predicted and target audio signals should be the same.")
    
    avg_result = {}
    avg_pesq = []
    avg_mse = []
    avg_consine = []
    si_snr = ScaleInvariantSignalNoiseRatio()
    
    for pred, gt in zip(preds, targets):
        # 1. Load audio
        pred_audio = load_audio(pred, sr=16000)
        gt_audio = load_audio(gt, sr=16000)
        
        N = min(len(pred_audio), len(gt_audio))
        pred_audio = pred_audio[:N]
        gt_audio = gt_audio[:N]
        
        # print(f"Pred audio: {pred_audio.shape}, gt audio: {gt_audio.shape}")
        
        
        # 2. Compute metrics
        mse_result = float(waveform_mse(pred_audio, gt_audio))
        cosine_result = mfcc_cosine_similarity(pred_audio, gt_audio)
        pesq_result = metric_pesq(torch.from_numpy(pred_audio), torch.from_numpy(gt_audio))
        
        si_snr.update(torch.from_numpy(pred_audio), torch.from_numpy(gt_audio))
        
        
        
        
        avg_pesq.append(pesq_result["pesq_wb"])
        avg_mse.append(mse_result)
        avg_consine.append(cosine_result)
        
    # compute the average result
    avg_result["pesq_wb"] = np.mean(avg_pesq)
    avg_result["mse"] = np.mean(avg_mse)
    avg_result["cosine"] = np.mean(avg_consine)
    avg_result["si_snr"] = si_snr.compute().item()
    return avg_result
    
    
class CLIPScore(nn.Module):
    def __init__(self, download_root=None, device='cpu', percentage=True):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device, jit=False, 
                                                     download_root=download_root)
        
        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(self.clip_model) # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.eval()
        self.clip_model.logit_scale.requires_grad_(False)
        self.percentage = percentage

    def score(self, prompts, image_paths):
        
        if not isinstance(prompts, list):
            prompts = [prompts, ]
            
        if not isinstance(image_paths, list):
            image_paths = [image_paths, ]
            
        txt_features = []    
        for prompt in prompts:
            # text encode
            text = clip.tokenize(prompt, truncate=True).to(self.device)
            txt_feature = F.normalize(self.clip_model.encode_text(text))
            txt_features.append(txt_feature)
            
        image_features = []
        # image encode
        for image_path in image_paths:
            pil_image = Image.open(image_path)
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_feature = F.normalize(self.clip_model.encode_image(image))
            image_features.append(image_feature)

        scores = []
        if len(txt_features) == 1 and len(txt_features) < len(image_features):
            txt_feature = txt_features[0]
            for image_feature in image_features:
                score = max(torch.sum(torch.mul(txt_feature, image_feature), dim=1, keepdim=True), torch.tensor(0))
                if self.percentage:
                    score = score.detach().cpu().item() * 100
                scores.append(score)
        elif len(image_features) == 1 and len(txt_features) > len(image_features):
            image_feature = image_features[0]
            for txt_feature in txt_features:
                # score
                # score = max((image_features * txt_feature).sum(axis=1), torch.tensor(0))
                score = max(torch.sum(torch.mul(txt_feature, image_feature), dim=1, keepdim=True), torch.tensor(0))
                if self.percentage:
                    score = score.detach().cpu().item() * 100
                scores.append(score)
        else:
            for txt_feature, image_feature in zip(txt_features, image_features):
                # score
                # score = max((image_features * txt_feature).sum(axis=1), torch.tensor(0))
                score = max(torch.sum(torch.mul(txt_feature, image_feature), dim=1, keepdim=True), torch.tensor(0))
                if self.percentage:
                    score = score.detach().cpu().item() * 100
                scores.append(score)
        return scores

    def score_same_modality(self, m1, m2, modality='text'):
        """
        Computing the same modality's similarity score (SS in the paper.).
        """
        features = [] 
        if modality == 'text':
            for prompt in [m1, m2]:
                # text encode
                text = clip.tokenize(prompt, truncate=True).to(self.device)
                txt_feature = F.normalize(self.clip_model.encode_text(text))
                features.append(txt_feature)
        elif modality == 'image':
            for image_path in [m1, m2]:
                pil_image = Image.open(image_path)
                image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                image_features = F.normalize(self.clip_model.encode_image(image))
                features.append(image_features)
                
        score = max(torch.sum(torch.mul(features[0], features[1]), dim=1, keepdim=True), torch.tensor(0))
        return score.detach().cpu().item()

    def inference_rank(self, prompt, generations_list):
        
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        txt_feature = F.normalize(self.clip_model.encode_text(text))
        
        txt_set = []
        img_set = []
        for generations in generations_list:
            # image encode
            img_path = generations
            pil_image = Image.open(img_path)
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_features = F.normalize(self.clip_model.encode_image(image))
            img_set.append(image_features)
            txt_set.append(txt_feature)
            
        txt_features = torch.cat(txt_set, 0).float() # [image_num, feature_dim]
        img_features = torch.cat(img_set, 0).float() # [image_num, feature_dim]
        rewards = torch.sum(torch.mul(txt_features, img_features), dim=1, keepdim=True)
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        
        return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()
 
 

def metrics_vision(preds, targets, device='cpu'):
    
    # preds_imgs = load_images_uint8(preds, device=device)
    # targets_imgs = load_images_uint8(targets, device=device)
    # result = metrics_is_fid(preds_imgs, targets_imgs, normalize=True)
    # tmp_dir = '~/.cache/know-bridger2/tmp'
    tmp_dir = './tmp-'+ str(np.random.randint(10000))
    try:
        preds = copy_resize_images(preds, os.path.join(tmp_dir, 'preds'), save_name_index=-4)
        targets = copy_resize_images(targets, os.path.join(tmp_dir, 'targets'), save_name_index=-1)
    
    
        result = torch_fidelity.calculate_metrics(
            input1=os.path.join(tmp_dir, 'preds'), 
            input2=os.path.join(tmp_dir, 'targets'),
            cuda=True, 
            isc=True, 
            fid=True, 
            kid=False, 
            prc=False, 
            verbose=False,
            samples_find_deep=False
        )
        
        clip_score = CLIPScore(percentage=False)
        clip_is = []
        for pred, target in zip(preds, targets):
            clip_i = clip_score.score_same_modality(pred, target, 'image')
            clip_is.append(clip_i)
    except Exception as e:
        shutil.rmtree(tmp_dir)
        raise e
        
    shutil.rmtree(tmp_dir)
    
    result['clip_i'] = np.mean(clip_is)
    return result
    


def metrics_f1_acc(preds, target):
    """
    Compute F1 and Average Precision scores for classification tasks.
    Args:
        preds: the predicted texts, shape (N,).
        target: the target texts, shape (N,).
    Returns:
        result: a dictionary containing the F1-macro, F1-micro, and accuracy scores.
        The keys are "f1_macro", "f1_micro", and "Acc".
        The value is the F1 or accuracy score, the higher the better.
    """
    if len(preds) != len(target):
        raise ValueError("The length of the predicted and target labels should be the same.")
    f1_macro = f1_score(target, preds, average="macro")
    f1_micro = f1_score(target, preds, average="micro")
    Acc = accuracy_score(target, preds)
    result = {"f1_macro": f1_macro, "Acc": Acc, "f1_micro": f1_micro}
    return result




   
    

if __name__ == "__main__":
    # img1_rgb = torch.randint(0, 155, (10, 3, 299, 299), dtype=torch.uint8)
    # img2_rbg = torch.randint(155, 255, (10, 3, 299, 299), dtype=torch.uint8)
    
    # img1_norm = torch.rand(10, 3, 299, 299)
    # img2_norm = torch.rand(10, 3, 299, 299)
    
    # rgb_result = metrics_is_fid(img1_rgb, img2_rbg)
    # print(f"RGB result: {rgb_result}")
    
    # norm_result = metrics_is_fid(img1_norm, img2_norm, normalize=True)
    # print(f"Normalized result: {norm_result}")

    # # For audio signals
    # preds = torch.randn(1, 25130)
    # target = torch.randn(1, 25130)
    # pesq_result = metric_pesq(preds, target)
    # print(f"PESQ result: {pesq_result}")
    
    # # For text
    # preds = ['the cat is on the mat']
    # target = [['there is a cat on the mat', 'a cat is on the mat']] 
    # text_result = metrics_language(preds, target)
    # print(f"Text result: {text_result}")
    
    # # For classification
    # preds = torch.randint(0, 5, (10,))
    # preds = [0, 1, 2, 0, 1, 2]
    # target = [0, 2, 1, 0, 0, 2]
    # f1_ap_result = metrics_f1_acc(preds, target)
    # print(f"Classification result: {f1_ap_result}")
    
    # preds = [
    #     '/mnt/share/keguanzhou/projects/know-bridger2/eval_dir/vggsound/audio-0.3/-QsPqh8Kyrc/completion/audio/prompt_0_0.wav',
    #     '/mnt/share/keguanzhou/projects/know-bridger2/eval_dir/vggsound/audio-0.3/0ut4if4B6zM/completion/audio/prompt_0_0.wav'
    # ]
    
    # targets = [
    #     '/mnt/share/keguanzhou/projects/MyDATA/vggsound/media/test/-QsPqh8Kyrc.wav',
    #     '/mnt/share/keguanzhou/projects/MyDATA/vggsound/media/test/0ut4if4B6zM.wav'
    # ]
    
    # # import json
    
    # # data = json.load(open('/mnt/share/keguanzhou/projects/know-bridger2/eval_dir/vggsound/audio-0.3/preds.json', 'r'))
    # # preds = data['preds']
    # # targets = data['targets']
    # metrics_audio_result = metrics_audio(preds, targets)
    # print(f"Audio result: {metrics_audio_result}")
    import json
    # json.dump(metrics_audio_result, open('metrics_audio.json', 'w'), indent=4)
    
    # import torch_fidelity
    
    
    
    # preds = [
    #     '/mnt/share/keguanzhou/projects/know-bridger2/eval_dir/vggsound/image-0.3/-1pRmoJIGQc/completion/image/prompt_0_0.png',
    #     '/mnt/share/keguanzhou/projects/know-bridger2/eval_dir/vggsound/image-0.3/-8C-gydUbR8/completion/image/prompt_0_0.png'
    # ]
    
    # targets = [
    #     '/mnt/share/keguanzhou/projects/MyDATA/vggsound/images/-1pRmoJIGQc/-1pRmoJIGQc_0.jpeg',
    #     '/mnt/share/keguanzhou/projects/MyDATA/vggsound/images/-8C-gydUbR8/-8C-gydUbR8_0.jpeg'
    # ]
    # import json
    '/mnt/share/keguanzhou/projects/know-bridger2/eval_dir/vggsound/flux1-baseline-image-0.3-lc/preds.json'
    preds_json = json.load(open('/mnt/share/keguanzhou/projects/know-bridger2/eval_dir/vggsound/baseline-qwen-vl-text-0.3/preds.json', 'r'))
    # preds_json = json.load(open('/mnt/share/keguanzhou/projects/know-bridger2/eval_dir/coco2014/flux1-baseline-image-0.3/preds.json', 'r'))
    
    preds = preds_json['preds']
    targets = preds_json['targets']
    
    metrics_language_result = metrics_language(preds, targets)
    print(f"Language result: {metrics_language_result}")
    
    # # 0.3 {'inception_score_mean': 7.879556730713543, 'inception_score_std': 1.0894332074475903, 'frechet_inception_distance': 174.83106000966566, 'precision': 0.8782051205635071, 'recall': 0.4038461446762085, 'f_score': 0.5532692206472158}
    # # 0.4 {'inception_score_mean': 10.694595743036185, 'inception_score_std': 1.1418339407847602, 'frechet_inception_distance': 151.08347283483988, 'precision': 0.8153846263885498, 'recall': 0.35769230127334595, 'f_score': 0.49725094161921685}
    
    # # copy_resize_and_cleanup_images(targets, './tmp/target', keep=True, save_name_index=-2)
    # # copy_resize_and_cleanup_images(preds, './tmp/preds', keep=True, save_name_index=-4)
    
    # metrics_dict = torch_fidelity.calculate_metrics(
    #     input1='/mnt/share/keguanzhou/projects/MyDATA/MSR-VTT/images', 
    #     input2='/mnt/share/keguanzhou/projects/MyDATA/MSR-VTT/images', 
    #     cuda=True, 
    #     isc=True, 
    #     fid=True, 
    #     kid=False, 
    #     prc=True, 
    #     verbose=False,
    #     samples_find_deep=True
    # )
    # print(metrics_dict)
    # metrics_vision_result = metrics_vision(preds, targets)
    # print(f"Vision result: {metrics_vision_result}")
    
    
    pass