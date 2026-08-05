# Colab 部署与训练

## 1. 启用 GPU

在 Colab 选择 `Runtime -> Change runtime type -> T4 GPU`，然后执行：

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

## 2. 获取项目

项目推送到 Git 仓库后：

```bash
!git clone <YOUR_REPOSITORY_URL> /content/Budgerigar
%cd /content/Budgerigar
!pip install -e .
!pytest -q
```

## 3. 下载 CMU ARCTIC

从 CMU ARCTIC 官方站点下载并解压至少两个真人库，例如 `bdl` 与 `slt`。把目录放在：

```text
/content/data/cmu_us_bdl_arctic/wav/arctic_a0001.wav
/content/data/cmu_us_slt_arctic/wav/arctic_a0001.wav
```

随后建立严格的同句配对清单：

```bash
!python -m budgerigar.prepare_arctic \
  --root /content/data \
  --output /content/Budgerigar/data/arctic \
  --target-speaker slt
```

## 4. 冒烟训练

```bash
!python -m budgerigar.train \
  --manifest data/arctic/train.jsonl \
  --steps 100 \
  --batch-size 2 \
  --token-dim 96 \
  --checkpoint checkpoints/smoke.pt \
  --device cuda
```

## 5. 正式基线

```bash
!python -m budgerigar.train \
  --manifest data/arctic/train.jsonl \
  --steps 30000 \
  --batch-size 4 \
  --segment-frames 320 \
  --checkpoint checkpoints/budgerigar.pt \
  --device cuda

!python -m budgerigar.evaluate \
  --manifest data/arctic/validation.jsonl \
  --checkpoint checkpoints/budgerigar.pt \
  --device cuda
```

## 6. 保存到 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!mkdir -p /content/drive/MyDrive/Budgerigar/checkpoints
!cp checkpoints/*.pt /content/drive/MyDrive/Budgerigar/checkpoints/
```

## 7. 诊断试听

```bash
!python -m budgerigar.stream_demo \
  --input /content/test.wav \
  --checkpoint checkpoints/budgerigar.pt \
  --output /content/output.wav \
  --device cuda
```

该命令用 Griffin-Lim 产生诊断音频，音质不会代表最终模型。确认内容学习有效后，再接入训练好的流式 HiFi-GAN/Vocos 声码器。

