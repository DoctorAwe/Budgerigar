from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import tempfile
import time

import torch

from .audio import AudioConfig, load_wave, log_mel
from .model import BudgerigarConfig, BudgerigarModel
from .stream_demo import approximate_waveform
from .train import choose_device


@lru_cache(maxsize=2)
def load_model(checkpoint_path: str, device_name: str) -> BudgerigarModel:
    device = choose_device(device_name)
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint}")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = BudgerigarModel(BudgerigarConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def visualization(source: torch.Tensor, prediction: torch.Tensor, state_energy: list[float]):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].imshow(source.T.numpy(), origin="lower", aspect="auto", cmap="magma")
    axes[0].set(title="输入语音 Log-Mel", ylabel="Mel bin")
    axes[1].imshow(prediction.T.numpy(), origin="lower", aspect="auto", cmap="viridis")
    axes[1].set(title="模型输出 Log-Mel（固定目标音色）", ylabel="Mel bin")
    axes[2].plot(state_energy, color="#21d4a7", linewidth=2, marker="o", markersize=3)
    axes[2].set(title="流式管线状态能量", xlabel="音频块", ylabel="RMS")
    for axis in axes:
        axis.grid(alpha=0.15)
    return figure


@torch.inference_mode()
def convert(audio_path: str | None, checkpoint_path: str, chunk_ms: int, device_name: str):
    if not audio_path:
        raise ValueError("请先上传音频或使用麦克风录音")
    model = load_model(checkpoint_path, device_name)
    device = next(model.parameters()).device
    audio = AudioConfig(n_mels=model.config.n_mels)
    waveform = load_wave(audio_path, audio.sample_rate)
    if waveform.numel() < audio.win_length:
        raise ValueError("录音过短，请至少录制 0.2 秒")
    features = log_mel(waveform, audio)
    chunk_frames = max(1, round(chunk_ms / 1000 * audio.sample_rate / audio.hop_length))

    state = None
    outputs = []
    state_energy = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, len(features), chunk_frames):
        chunk = features[None, start:start + chunk_frames].to(device)
        prediction, state = model.forward_chunk(chunk, state)
        outputs.append(prediction[0].cpu())
        stacked = torch.stack(state.layers)
        state_energy.append(float(stacked.float().square().mean().sqrt().cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started

    prediction = torch.cat(outputs)
    generated = approximate_waveform(prediction, audio, phase_reference=waveform)
    output_dir = Path(tempfile.mkdtemp(prefix="budgerigar_demo_"))
    output_path = output_dir / "budgerigar_output.wav"
    import torchaudio
    torchaudio.save(str(output_path), generated[None].cpu(), audio.sample_rate)

    duration = waveform.numel() / audio.sample_rate
    realtime_factor = inference_seconds / max(duration, 1e-6)
    metrics = (
        f"### 推理结果\n"
        f"- 输入时长：**{duration:.2f} 秒**\n"
        f"- 分块大小：**{chunk_ms} ms**（{chunk_frames} Mel 帧）\n"
        f"- 模型推理：**{inference_seconds:.3f} 秒**\n"
        f"- 实时率 RTF：**{realtime_factor:.3f}**（低于 1 表示快于实时）\n"
        f"- 流式块数：**{len(outputs)}**\n"
        f"- 设备：**{device}**"
    )
    return str(output_path), visualization(features.cpu(), prediction, state_energy), metrics


def validation_examples(manifest_path: str, limit: int) -> dict[str, dict[str, str]]:
    manifest = Path(manifest_path)
    if not manifest.exists():
        return {}
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = {}
    for row in rows[:limit]:
        label = f'{row["source_speaker"]} → {row["target_speaker"]} · {row["utterance_id"]}'
        examples[label] = row
    return examples


def build_app(
    default_checkpoint: str,
    default_device: str,
    validation_manifest: str = "data/arctic/validation.jsonl",
    max_examples: int = 24,
):
    import gradio as gr

    examples = validation_examples(validation_manifest, max_examples)

    def select_example(label: str):
        if not label or label not in examples:
            raise ValueError("请选择一个验证集样本")
        row = examples[label]
        description = (
            f'验证样本：**{row["utterance_id"]}**　'
            f'输入说话人：**{row["source_speaker"]}**　'
            f'目标说话人：**{row["target_speaker"]}**'
        )
        return row["source_path"], row["target_path"], description

    with gr.Blocks(title="Budgerigar 鹦鹉学舌演示", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Budgerigar · 鹦鹉学舌流式演示\n"
            "上传或录制一句英语，模型会用训练目标音色进行声学复述，并展示流式内部状态。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                if examples:
                    with gr.Group():
                        example = gr.Dropdown(
                            choices=list(examples), label="数据集验证样本",
                            info="选择一条未参与训练的源真人语音",
                        )
                        load_example = gr.Button("载入验证样本")
                        example_info = gr.Markdown("从验证集选择样本，或在下方上传自己的录音。")
                audio_input = gr.Audio(
                    label="输入真人语音", sources=["upload", "microphone"], type="filepath"
                )
                checkpoint = gr.Textbox(label="Checkpoint", value=default_checkpoint)
                with gr.Row():
                    chunk_ms = gr.Slider(80, 1000, value=320, step=40, label="流式分块（ms）")
                    device = gr.Dropdown(["auto", "cuda", "cpu"], value=default_device, label="设备")
                run = gr.Button("开始模仿", variant="primary")
                gr.Markdown("提示：当前使用输入相位辅助诊断合成；可懂度可用于检查模型，但最终音色仍需神经声码器。")
            with gr.Column(scale=1):
                target_reference = gr.Audio(label="目标真人参考（验证集）", interactive=False)
                audio_output = gr.Audio(label="模型复述输出")
                metrics = gr.Markdown("等待输入…")
        plot = gr.Plot(label="声学与状态可视化")
        run.click(
            fn=convert,
            inputs=[audio_input, checkpoint, chunk_ms, device],
            outputs=[audio_output, plot, metrics],
        )
        if examples:
            load_example.click(
                fn=select_example,
                inputs=[example],
                outputs=[audio_input, target_reference, example_info],
            )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Budgerigar Gradio demo")
    parser.add_argument("--checkpoint", default="checkpoints/budgerigar.pt")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--manifest", default="data/arctic/validation.jsonl")
    parser.add_argument("--max-examples", type=int, default=24)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio URL")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app = build_app(args.checkpoint, args.device, args.manifest, args.max_examples)
    allowed_paths = [str(Path(args.manifest).expanduser().resolve().parent)]
    examples = validation_examples(args.manifest, args.max_examples)
    for row in examples.values():
        allowed_paths.extend([str(Path(row["source_path"]).parent), str(Path(row["target_path"]).parent)])
    app.launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share,
        show_error=True, allowed_paths=sorted(set(allowed_paths)),
    )


if __name__ == "__main__":
    main()
