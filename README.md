# Budgerigar / 鹦鹉学舌模型

Budgerigar 是一个因果流式的语音复述基线：输入真人语音的 Mel 流，模型使用固定目标音色重建相同内容。当前里程碑聚焦 CMU ARCTIC 的“同句、不同真人”平行训练。

```text
waveform -> log-Mel -> causal token pipeline -> target log-Mel -> waveform
```

## 快速开始

```bash
pip install -e .
python -m budgerigar.prepare_arctic --root /path/to/cmu_arctic --output data/arctic
python -m budgerigar.train --manifest data/arctic/train.jsonl --device cuda
python -m budgerigar.evaluate --manifest data/arctic/validation.jsonl \
  --checkpoint checkpoints/budgerigar.pt --device cuda
```

详细的 Colab 流程见 [COLAB_TRAINING.md](COLAB_TRAINING.md)，设计与阶段边界见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。

