# AFM2

Official repository for **"How Far Are We from Generating Missing Modalities with Foundation Models?"**

AFM2 provides a unified interface for evaluating and deploying missing modality generation pipelines using foundation models such as **Qwen2.5** and **Qwen2.5-Omni**.

---

## 🚀 Quick Start

### ✅ Step 1: Install AFM2 and Dependencies

**Note:** This step will take a lot of time, so grab a cup of coffee. 😋

```bash
git clone https://github.com/Guanzhou-Ke/AFM2.git
cd AFM2
conda env create -f environment.yaml
conda activate afm2
pip install -e .
```

You can add the following line to your README to guide users on resolving the `Qwen2_5Omni` import issue:

---

> ⚠️ **Note:** If `transformers` cannot find `Qwen2_5Omni`, install the official Qwen dependencies using:

```bash
pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
```

This will install the latest implementation of Qwen2.5 models, including `Qwen2_5Omni`, into your environment.


> Install `torch-fidelity`

```
cd torch-fidelity
pip install -e .
```

---

### ✅ Step 2: Start the Generator Server

```bash
python servers/generator_server.py --port 12399
```

> **Optional Test:** Use the following commands to verify the generator server is working.

#### 🖼️ Image Generation

```bash
curl -X POST http://0.0.0.0:12399/image_generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a magical forest in the moonlight",
    "output_path": "./outputs",
    "save_name": "prompt_0"
}'
```

📍 Output path: `./outputs/completion/image/prompt_0_0.png`

#### 🔊 Audio Generation

```bash
curl -X POST http://0.0.0.0:12399/audio_generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a magical forest in the moonlight",
    "output_path": "./outputs",
    "save_name": "prompt_0"
}'
```

📍 Output path: `./outputs/completion/audio/prompt_0_0.wav`

> Configuration file: `afm2/servers/generator_config.yaml`

---

### ✅ Step 3: Launch vLLM Servers for Qwen Models

#### (A) Install `vllm`

We recommend setting up `vllm` in a separate virtual environment.
Refer to the official [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html).

#### (B) Start the Model Servers

```bash
# Qwen2.5-14B-Instruct-1M on port 12377
CUDA_VISIBLE_DEVICES=4 vllm serve Qwen/Qwen2.5-14B-Instruct-1M \
  --dtype auto \
  --api-key token-abc123 \
  --max_model_len 16384 \
  --port 12377 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --disable-log-stats

# Qwen2.5-Omni-7B on port 12388
CUDA_VISIBLE_DEVICES=5 vllm serve Qwen/Qwen2.5-Omni-7B \
  --dtype bfloat16 \
  --api-key token-abc123 \
  --max_model_len 16384 \
  --port 12388 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --disable-log-stats
```

> LLM agent behavior can be configured in:
> `afm2/configs/mainagent.yaml`

---

### ✅ Step 4: Test the LLM Servers

#### 🧪 Qwen2.5-Omni (Port 12388)

```bash
curl http://localhost:12388/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-abc123" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/cough.wav"}},
        {"type": "text", "text": "What is the text in the illustration and what is the sound in the audio?"}
      ]}
    ]
}'
```

#### 🧪 Qwen2.5-14B-Instruct (Port 12399)

```bash
curl http://localhost:12399/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-abc123" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "text", "text": "Hello, give me a joke."}
      ]}
    ]
}'
```

---

## 🔍 Inference Pipeline

### ✅ Step 5: Prepare Dataset and Configuration

1. Prepare your dataset and simulated missing modality files.
2. Configure your experiment using:

* Dataset configs: `afm2/configs/eval/`
* Missing modality simulation: `afm2/configs/missing-info/`

Each `.yaml` file includes parameters like input paths, modality control, and output directory (`work_dir`).

---

### ✅ Step 6: Run Evaluation

```bash
python afm2/eval.py --dataset [dataset_name]
```

Example:

```bash
python afm2/eval.py --dataset vggsound
```

📁 Results are saved to the `work_dir` specified in your config.

---

### ✅ Step 7: Compute Metrics

AFM2 will automatically compute and log key evaluation metrics upon completion, including:

* **Fidelity** (e.g., FID, PESQ)
* **Semantic alignment** (e.g., CLIP similarity, Match Error Rate)

No additional manual evaluation steps are needed.


## Citation

Please consider citing our work if you find it beneficial to your research.
```
@article{ke2025knowledge,
  title={Knowledge bridger: Towards training-free missing multi-modality completion},
  author={Ke, Guanzhou and He, Shengfeng and Wang, Xiao Li and Wang, Bo and Chao, Guoqing and Zhang, Yuanyang and Xie, Yi and Su, HeXing},
  journal={arXiv preprint arXiv:2502.19834},
  year={2025}
}

@article{ke2025far,
  title={How Far Are We from Predicting Missing Modalities with Foundation Models?},
  author={Ke, Guanzhou and Xie, Yi and Wang, Xiaoli and Chao, Guoqing and Wang, Bo and He, Shengfeng},
  journal={arXiv preprint arXiv:2506.03530},
  year={2025}
}
```
