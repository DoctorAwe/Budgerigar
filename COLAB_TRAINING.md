# Colab 部署与训练

## 1. 启用 GPU

在 Colab 选择 `Runtime -> Change runtime type`。Colab 会根据套餐、计算单元余额和实时库存动态提供 GPU，不能保证每次都出现同一种型号。

### GPU 对比与本项目建议

| GPU | 常见显存 | 混合精度 | 对当前基线的定位 |
|---|---:|---|---|
| T4 | 16 GB | FP16 Tensor Core | **当前首选**；足够运行 `batch-size=4`、`token-dim=192` 的首版模型 |
| L4 | 24 GB | FP16/BF16 Tensor Core | **最佳升级**；可以提高 batch、缩短训练时间，并给后续内容编码器留显存 |
| V100 | 常见为 16 GB | FP16 Tensor Core | 可用，但架构较旧；通常选到就直接训练，不必为了 T4 重连 |
| P100 | 16 GB | FP16，但无 Tensor Core | 能训练，混合精度收益较弱；速度通常不如 T4/L4 |
| A100 | Colab 常见为 40 GB | TF32/FP16/BF16 Tensor Core | 适合后续 HuBERT/WavLM、神经声码器或大 batch；当前小基线利用率不足 |

阶段建议：

- 数据下载、解压、manifest 生成：使用 CPU，避免浪费 GPU 计算单元；
- `--steps 100` 冒烟训练：T4；
- 当前 30k step 正式基线：优先 T4，想明显缩短时间则选择 L4；
- 加入 HuBERT/WavLM 或训练神经声码器：优先 L4，显存不足再用 A100；
- TPU：当前代码是 PyTorch CUDA 路径且包含逐时间步状态，不建议在本阶段使用。

当前模型有串行的 16 层时间状态更新，数据加载与 Log-Mel 也在 CPU 上实时完成，因此从 T4 升到 A100 不会按理论算力成比例加速。正式使用 A100 前，应先缓存 Mel 特征，并对时间循环做编译或结构优化。

连接运行时后执行以下单元格确认实际硬件与显存：

```python
import torch

if not torch.cuda.is_available():
    raise RuntimeError("当前运行时没有 GPU，请在 Runtime -> Change runtime type 中启用 GPU")

properties = torch.cuda.get_device_properties(0)
print("GPU:", properties.name)
print("显存:", f"{properties.total_memory / 2**30:.1f} GiB")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
!nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

若 T4 上出现 CUDA OOM，依次把 `--batch-size` 从 4 降到 2、把 `--segment-frames` 从 320 降到 240；不要先缩减模型层数，否则会改变需要验证的时间状态结构。

## 2. 获取项目

项目推送到 Git 仓库后：

```bash
!git clone https://github.com/DoctorAwe/Budgerigar /content/Budgerigar
%cd /content/Budgerigar
!pip install -e .
!pytest -q
```

## 3. 自动下载并处理 CMU ARCTIC

下面的单个 Python 单元格会完成以下工作：

1. 从 CMU FestVox 官方站点下载 `bdl`（男声）和 `slt`（女声）；
2. 支持断点式重复运行：已有且可正常打开的压缩包不会重新下载；
3. 安全解压 `.tar.bz2`，拒绝写出数据目录的归档成员；
4. 检查 WAV 数量、声道数、采样率和样本宽度；
5. 调用项目的数据处理程序，生成严格按句子 ID 配对的训练和验证 manifest；
6. 打印配对数量和若干样例供人工检查。

官方数据约有 1132 条音素平衡语句。初次运行需要下载约 200 MB，具体大小可能随官方文件版本变化。

```python
from pathlib import Path
import json
import hashlib
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import wave

PROJECT_ROOT = Path("/content/Budgerigar")
DOWNLOAD_ROOT = Path("/content/downloads/cmu_arctic")
DATA_ROOT = Path("/content/data/cmu_arctic")
MANIFEST_ROOT = PROJECT_ROOT / "data/arctic"

# 首版固定 slt 为输出音色，bdl 为输入真人。可以向 SOURCES 加入
# clb/rms/jmk/ksp；加入其他说话人前请先核对官方许可。
TARGET_SPEAKER = "slt"
SOURCES = ["bdl"]
# SHA-256 来自 torchaudio 的 CMUARCTIC 数据集实现。
CHECKSUMS = {
    "bdl": "26b91aaf48b2799b2956792b4632c2f926cd0542f402b5452d5adecb60942904",
    "slt": "7c173297916acf3cc7fcab2713be4c60b27312316765a90934651d367226b4ea",
    "clb": "3f16dc3f3b97955ea22623efb33b444341013fc660677b2e170efdcc959fa7c6",
    "rms": "c6dc11235629c58441c071a7ba8a2d067903dfefbaabc4056d87da35b72ecda4",
    "jmk": "3a37c0e1dfc91e734fdbc88b562d9e2ebca621772402cdc693bbc9b09b211d73",
    "ksp": "8029cafce8296f9bed3022c44ef1e7953332b6bf6943c14b929f468122532717",
}

DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)


def archive_name(speaker: str) -> str:
    return f"cmu_us_{speaker}_arctic.tar.bz2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_archive(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False
    try:
        with tarfile.open(path, "r:bz2") as archive:
            structure_ok = any(member.name.endswith("/wav/arctic_a0001.wav") for member in archive)
        speaker = path.name.removeprefix("cmu_us_").removesuffix("_arctic.tar.bz2")
        expected_hash = CHECKSUMS.get(speaker)
        return structure_ok and (expected_hash is None or sha256(path) == expected_hash)
    except tarfile.TarError:
        return False


def download(speaker: str, retries: int = 3) -> Path:
    name = archive_name(speaker)
    destination = DOWNLOAD_ROOT / name
    if valid_archive(destination):
        print(f"[cache] {name} ({destination.stat().st_size / 2**20:.1f} MiB)")
        return destination
    destination.unlink(missing_ok=True)
    # 新版 torchaudio 使用 /packed/ 下无版本号的归档。FestVox 对部分
    # Colab 出口的 HTTPS 支持不稳定，因此依次尝试 HTTPS、HTTP 和无 www 主机。
    urls = [
        f"https://www.festvox.org/cmu_arctic/packed/{name}",
        f"http://www.festvox.org/cmu_arctic/packed/{name}",
        f"https://festvox.org/cmu_arctic/packed/{name}",
        f"http://festvox.org/cmu_arctic/packed/{name}",
    ]
    failures = []
    for url in urls:
        for attempt in range(1, retries + 1):
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.unlink(missing_ok=True)
            try:
                print(f"[download {attempt}/{retries}] {url}")
                request = urllib.request.Request(url, headers={"User-Agent": "Budgerigar-Colab/0.1"})
                with urllib.request.urlopen(request, timeout=90) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                temporary.replace(destination)
                if not valid_archive(destination):
                    raise RuntimeError("归档结构或 SHA-256 校验失败")
                print(f"[verified sha256] {name}")
                return destination
            except Exception as error:
                temporary.unlink(missing_ok=True)
                destination.unlink(missing_ok=True)
                failures.append(f"{url}: {type(error).__name__}: {error}")
                if attempt < retries:
                    time.sleep(2 ** attempt)
                else:
                    print(f"[mirror failed] {url}: {error}")
    details = "\n".join(failures)
    raise RuntimeError(f"所有 FestVox 下载地址均失败：\n{details}")


def safe_extract(path: Path) -> None:
    directory_name = path.name.removesuffix(".tar.bz2")
    expected = DATA_ROOT / directory_name
    wav_dir = expected / "wav"
    if wav_dir.exists() and len(list(wav_dir.glob("arctic_*.wav"))) >= 1000:
        print(f"[cache] {expected.name} already extracted")
        return
    root = DATA_ROOT.resolve()
    with tarfile.open(path, "r:bz2") as archive:
        for member in archive.getmembers():
            destination = (DATA_ROOT / member.name).resolve()
            if root != destination and root not in destination.parents:
                raise RuntimeError(f"不安全的归档路径: {member.name}")
        archive.extractall(DATA_ROOT, filter="data")
    print(f"[extract] {expected}")


def inspect_speaker(speaker: str) -> None:
    wav_dir = DATA_ROOT / f"cmu_us_{speaker}_arctic" / "wav"
    files = sorted(wav_dir.glob("arctic_*.wav"))
    if len(files) < 1000:
        raise RuntimeError(f"{speaker}: 只找到 {len(files)} 个 WAV，数据可能不完整")
    with wave.open(str(files[0]), "rb") as handle:
        metadata = (handle.getframerate(), handle.getnchannels(), handle.getsampwidth())
    if metadata != (16_000, 1, 2):
        raise RuntimeError(f"{speaker}: 意外 WAV 格式 {metadata}，预期 (16000, 1, 2)")
    print(f"[verify] {speaker}: {len(files)} WAV, 16 kHz mono PCM16")


speakers = list(dict.fromkeys(SOURCES + [TARGET_SPEAKER]))
for speaker in speakers:
    safe_extract(download(speaker))
    inspect_speaker(speaker)

command = [
    sys.executable, "-m", "budgerigar.prepare_arctic",
    "--root", str(DATA_ROOT),
    "--output", str(MANIFEST_ROOT),
    "--target-speaker", TARGET_SPEAKER,
]
subprocess.run(command, cwd=PROJECT_ROOT, check=True)

for split in ("train", "validation"):
    manifest = MANIFEST_ROOT / f"{split}.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    print(f"[manifest] {split}: {len(rows)} pairs -> {manifest}")
    for row in rows[:2]:
        print(" ", row["source_speaker"], "->", row["target_speaker"], row["utterance_id"])
```

成功运行后目录应类似：

```text
/content/data/cmu_arctic/cmu_us_bdl_arctic/wav/arctic_a0001.wav
/content/data/cmu_arctic/cmu_us_slt_arctic/wav/arctic_a0001.wav
/content/Budgerigar/data/arctic/train.jsonl
/content/Budgerigar/data/arctic/validation.jsonl
```

重新执行该单元格不会重复下载完整文件。若上一次下载中断，残留的 `.part` 文件会被清理后重新下载。

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

## 8. 启动 Web 可视化演示

先安装演示依赖：

```bash
!pip install -q -r requirements-demo.txt
```

然后启动 Gradio。`--share` 会生成一个临时公网链接，可在电脑或手机浏览器中打开：

```bash
!python -m budgerigar.web_demo \
  --checkpoint checkpoints/budgerigar.pt \
  --device cuda \
  --share
```

页面支持：

- 上传 WAV/MP3 或直接使用麦克风录音；
- 调整 80～1000 ms 流式分块；
- 试听模型复述结果；
- 对比输入和输出 Log-Mel 频谱；
- 观察 16 层因果管线的状态能量；
- 查看推理时间、实时率和流式块数。

Colab 单元格会持续运行以维持 Web 服务。停止单元格或断开运行时后，临时公网链接会失效。不要公开分享包含私人录音或敏感 checkpoint 的演示链接。
