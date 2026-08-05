# Budgerigar / 鹦鹉学舌模型

Budgerigar 的当前主线是统一的因果流式波形处理器：持续接收真人时域音频，并持续返回等长时域音频。静音、开始复述、内容、音色和结束全部由模型输出波形表达，不使用外部 VAD、发声动作或 Griffin-Lim。

```text
waveform -> causal encoder -> recurrent/attention memory -> internal waveform decoder -> waveform
```

## 快速开始

```bash
pip install -e .
python -m budgerigar.prepare_arctic --root /path/to/cmu_arctic --output data/arctic
python -m budgerigar.train_waveform --manifest data/arctic_multi/train.jsonl --device cuda
python -m budgerigar.evaluate_waveform --manifest data/arctic_multi/validation.jsonl \
  --checkpoint checkpoints/budgerigar_waveform.pt --device cuda
```

详细的 Colab 流程见 [COLAB_TRAINING.md](COLAB_TRAINING.md)，设计与阶段边界见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。
