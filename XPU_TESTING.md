# Intel GPU (XPU) 测试指南

本文用于在 Intel GPU 上验证本分支（`claude/awesome-galileo-ibp4tf`）的 XPU 适配。代码改动和已知限制见 `docs/zh/Pipeline_Usage/GPU_support.md` 中的 “Intel GPU (XPU)” 一节。

测试分四步，从不下载模型的快速检查开始，逐步扩大范围。前一步通过后再做下一步，出问题时更容易定位。

> 命令行说明：以下命令以 Linux bash 为准。Windows 上建议使用 Git Bash 或 WSL 运行（`sed` 和行尾 `\` 续行可直接使用）；若使用 PowerShell，需把续行符 `\` 换成反引号 `` ` ``。

## 第 0 步：安装环境

1. 按照 PyTorch 官方文档 [Getting Started on Intel GPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) 安装 Intel GPU 驱动。
2. **先**安装 XPU 版 PyTorch，再安装 DiffSynth-Studio。顺序不能反，否则 `pip install -e .` 会从 PyPI 安装 CUDA 版 torch。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
```
```bash
git clone -b claude/awesome-galileo-ibp4tf https://github.com/YzyParry/DiffSynth-Studio.git
```
```bash
cd DiffSynth-Studio && pip install -e .
```

3. 确认 torch 是 XPU 版本且能识别到显卡：

```bash
python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
```

预期输出的版本号带 `+xpu`，并且是 `True`。如果版本号变成了 `+cu…`，说明 torch 被替换成了 CUDA 版，重新执行第一条安装命令即可。

## 第 1 步：冒烟测试（几分钟，不下载模型）

```bash
python xpu_smoke_test.py
```

脚本逐项检查，单项失败不影响其他项，最后打印 PASS / FAIL / WARN / SKIP 汇总表：

| 检查项 | 对应的功能 |
| --- | --- |
| diffsynth with XPU support is imported | 导入的是本分支的代码 |
| device auto-detection | `get_device_type()`、Pipeline 默认设备、通信后端 `xccl` |
| memory query | `vram_limit` 和显存管理用到的 `torch.xpu.mem_get_info` |
| bf16 matmul / attention | 基础计算；注意力走 PyTorch SDPA |
| float64 basic ops / float64 / complex128 RoPE | **FP64 支持**，Wan 等模型的 RoPE 需要 |
| autocast-disabled fp32 regions | 本次修复的 fp32 区域在 XPU autocast 下生效 |
| seeded random generator on device | `rand_device="xpu"` 的可复现性 |
| VRAM management | CPU offload 的层按需加载到 XPU 上计算 |
| pinned memory + non-blocking copy + synchronize | 训练 CPU offload 依赖的机制 |
| torch.compile（可选） | Inductor / Triton XPU |
| XCCL backend available（可选） | 多卡训练 |

重点关注两处：

- 开头设备信息里的 `native fp64`。
- 两项 float64 检查。如果显卡没有原生 FP64（例如 Arc A 系列、核显），Wan 等使用 `float64` RoPE 的模型很可能在这里出问题。

在没有 GPU 的机器上可以用 `SMOKE_DEVICE=cpu python xpu_smoke_test.py` 空跑，用来检查脚本本身。

## 第 2 步：真实推理

示例脚本中与 CUDA 相关的写法只有 `"cuda"` 和 `torch.cuda.mem_get_info`，用 `sed` 替换即可得到 XPU 版本，不需要修改原文件：

```bash
mkdir -p xpu_tests && for f in examples/wanvideo/model_inference_low_vram/Wan2.1-T2V-1.3B.py examples/z_image/model_inference_low_vram/Z-Image-Turbo.py; do sed -e 's/"cuda"/"xpu"/g' -e 's/torch\.cuda\./torch.xpu./g' "$f" > "xpu_tests/$(basename "$f")"; done
```
```bash
python xpu_tests/Wan2.1-T2V-1.3B.py
```
```bash
python xpu_tests/Z-Image-Turbo.py
```

（`xpu_tests/` 下是生成的临时脚本，不需要提交。）

| 模型 | 覆盖的路径 |
| --- | --- |
| Wan2.1-T2V-1.3B | FP64 RoPE、磁盘 offload 和显存管理、T5 文本编码器、分块 VAE、视频转视频 |
| Z-Image-Turbo | 图像 Pipeline、Qwen3 文本编码器、`rand_device="xpu"` |

通过标准：能完整跑完，输出不是全黑或纯噪声，并记录耗时。

- **与 CUDA 对比**：Wan 默认 `rand_device="cpu"`，初始噪声和设备无关，所以同一 seed 的 XPU 结果可以直接和 CUDA 结果对比。Z-Image 的噪声在设备上生成，不能逐像素对比。
- **显存充足时**：`model_inference_low_vram` 使用磁盘 offload，速度较慢。可以换成 `examples/*/model_inference/` 下的对应脚本，同样用 `sed` 转换。
- **其他模型**：任意示例脚本都可以用同样的 `sed` 命令转换。

## 第 3 步：训练

下载示例数据集：

```bash
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "wanvideo/Wan2.1-T2V-1.3B/*" --local_dir ./data/diffsynth_example_dataset
```

单卡短程 LoRA 训练（帧数和步数已调小，并开启梯度检查点）：

```bash
accelerate launch --num_processes 1 examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B/metadata.csv \
  --height 480 --width 832 --num_frames 17 --dataset_repeat 5 --num_epochs 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 --remove_prefix_in_ckpt "pipe.dit." \
  --output_path ./models/train/Wan2.1-T2V-1.3B_lora_xpu \
  --lora_base_model dit --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32 \
  --use_gradient_checkpointing
```

通过标准：训练完成，`--output_path` 下生成 LoRA 权重文件。然后依次测试以下变体：

1. **CPU offload 训练**：在上面的命令末尾加 `--enable_model_cpu_offload`。这条路径原来调用 `torch.cuda.synchronize()`，在 XPU 上会直接崩溃，是本次修复的重点。
2. **多卡训练**（有多张卡时）：把 `--num_processes 1` 改为 `--multi_gpu --num_processes 2`，验证 `xccl` 通信。
3. **验证 LoRA 效果**：用第 2 步的 `sed` 命令转换 `examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-1.3B.py`，生成视频检查效果。

显存不足时，可以降低 `--height` / `--width`（需为 16 的倍数），或加上 `--use_gradient_checkpointing_offload`。

## 第 4 步：记录结果

以下信息用于提交上游 PR，也便于定位问题：

```text
显卡型号 / 数量：
驱动版本：
操作系统：
torch 版本（含 +xpu 后缀）：

冒烟测试汇总表：
（粘贴 xpu_smoke_test.py 的输出）

推理：
- Wan2.1-T2V-1.3B：通过 / 失败，耗时，输出是否正常
- Z-Image-Turbo：通过 / 失败，耗时，输出是否正常

训练：
- 单卡 LoRA：通过 / 失败，耗时
- --enable_model_cpu_offload：通过 / 失败
- 多卡 xccl：通过 / 失败 / 未测

CUDA 回归（如有 CUDA 机器，用未修改的原版示例脚本）：通过 / 失败 / 未测
```

如有失败，请保留完整的报错信息（traceback）。
