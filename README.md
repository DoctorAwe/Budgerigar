# Budgerigar（鹦鹉）实时语音复述模型

Budgerigar 的目标是：持续监听任意说话人的语音流，保留其语言内容和必要的节奏线索，并用模型固定的“自身音色与表达习惯”低延迟复述。它不是录音回放、目标说话人克隆或语音识别后再 TTS 的简单拼接。

当前仓库处于设计阶段。完整路线见 [docs/00_项目路线图.md](docs/00_项目路线图.md)。

> 运行环境约束：本地工作区只用于代码与文档开发、静态检查和轻量单元测试。数据集下载、数据真实性检查、GPU 训练、真实音频评测和实时性能验证统一在 Google Colab 中执行。Colab 结果是项目验收依据。

## 当前可运行能力

- `AudioConfig`：统一 24 kHz、tick 和 look-ahead 时序配置。
- `AudioChunker`：把浏览器/声卡的不规则输入块整理成固定模型 tick，并支持 flush/reset。
- `ManifestRecord`：读取阶段 1 定义的 JSONL 数据清单。
- `budgerigar-audit`：检查音频路径、时长、重复 ID、平行配对和切分泄漏。

本地开发环境可运行不依赖真实数据和 GPU 的检查：

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
budgerigar-audit data/manifest.jsonl
```

其中 `budgerigar-audit` 的真实数据报告应在 Colab 下载并整理数据集后生成。

推进状态见 [docs/09_实施进度.md](docs/09_实施进度.md)。

Colab 数据入口：[notebooks/Budgerigar_Data.ipynb](notebooks/Budgerigar_Data.ipynb)。它会下载并校验 CMU ARCTIC、索引用户已按官方研究条款取得的 ESD、运行真实 manifest 审计并保存运行元数据。

Colab 特征入口：[notebooks/Budgerigar_Features.ipynb](notebooks/Budgerigar_Features.ipynb)。在数据审计通过后生成可恢复的 24 kHz log-Mel、能量和 VAD 缓存。

Colab 连续神经复读基线：[notebooks/Budgerigar_Neural_Echo_Train.ipynb](notebooks/Budgerigar_Neural_Echo_Train.ipynb)。模型在同一连续时间轴上学习先听完整表达、保持静默，再用固定声线复读；不存在程序化的监听/结束/朗读状态机。

Colab 时间轴行为评估：[notebooks/Budgerigar_Neural_Echo_Evaluate.ipynb](notebooks/Budgerigar_Neural_Echo_Evaluate.ipynb)。在扩大训练前检查提前发声、思考间隔、复读召回和全静默退化。

Colab 内容保持评估：[notebooks/Budgerigar_Neural_Echo_Content_Evaluate.ipynb](notebooks/Budgerigar_Neural_Echo_Content_Evaluate.ipynb)。通过正确/打乱目标检索与输入消融，检查神经记忆是否真正保留当前句子。

Colab 层级 token 记忆训练：[notebooks/Budgerigar_Hierarchical_Echo_Train.ipynb](notebooks/Budgerigar_Hierarchical_Echo_Train.ipynb)。针对第一代内容检索失败，引入 AutoMachine 式 token bank、注意力读取和打乱目标对比损失。

Colab 层级模型联合评估：[notebooks/Budgerigar_Hierarchical_Echo_Evaluate.ipynb](notebooks/Budgerigar_Hierarchical_Echo_Evaluate.ipynb)。同一 checkpoint 必须同时通过时间轴行为和内容保持评估。

Colab 内容记忆预训练：[notebooks/Budgerigar_Content_Memory_Train.ipynb](notebooks/Budgerigar_Content_Memory_Train.ipynb)。在重新训练声学复读前，用 transcript CTC 与音频—文本 InfoNCE 强制 token bank 保存可恢复语言内容。

## 文档导航

1. [阶段 0：问题定义与验收标准](docs/01_阶段0_问题定义.md)
2. [阶段 1：数据集与数据管线](docs/02_阶段1_数据集.md)
3. [阶段 2：离线基线模型](docs/03_阶段2_离线基线.md)
4. [阶段 3：AutoMachine 式层级记忆](docs/04_阶段3_层级记忆.md)
5. [阶段 4：流式模型与实时推理](docs/05_阶段4_流式推理.md)
6. [阶段 5：训练课程与损失函数](docs/06_阶段5_训练.md)
7. [阶段 6：评估与消融实验](docs/07_阶段6_评估.md)
8. [阶段 7：Colab、Gradio 与交付](docs/08_阶段7_部署演示.md)
9. [实施进度](docs/09_实施进度.md)

## 设计原则

- 内容可变，输出身份固定：输入音色不能泄漏到输出。
- 先离线验证可学性，再施加因果和延迟约束。
- 每个阶段都必须有可量化的退出条件。
- 层级 token 是持续更新的状态，不是固定长度缓存。
- 第一版优先做可解释、可调试的声学特征路线，验证后再尝试神经音频 codec。
