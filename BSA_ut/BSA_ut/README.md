# BSA attention benchmark

独立测试三种 mask 的前向、反向性能，默认 BF16、BSHD `[1, 255424, 4, 128]`。

| 用例 | 实现 | Mask |
| --- | --- | --- |
| `causal` | FlashAttention `magi_backend` | 标准 token causal：`k <= q` |
| `bsa-causal` | cuDNN BSA forward；可选通用 blk64 或专用 Flex backward | 64-token block causal，对角块内全可见 |
| `production` | cuDNN BSA | 四样本 text/video/audio packed mask，视频块 Top-K |

两个 causal 用例的 mask 不完全相同，耗时比不是严格同一算子的加速比。

## 环境与安装

统一锁频测试面向 **GB200 / SM100**。基础环境需提供 **Python 3.13.12、Torch 2.10、
CUDA 13 toolkit、C++ 编译器、pip、Git**，并能访问 GitHub 和包下载源。

用已有 Torch 的默认 Python 创建两个环境，再分别安装：

```bash
cd BSA_ut
python create_envs.py .venv/bsa .venv/causal

source .venv/bsa/bin/activate
bash install.sh
deactivate

source .venv/causal/bin/activate
bash install_causal.sh
deactivate
```

两个环境共享基础环境的 Torch，无需重新下载；CUTLASS 等依赖分别安装。
基础环境需保留且在测试期间保持不变。环境目录须尚未存在；更换目录后同步调整运行路径。

| 环境 | 固定源码 | 主要依赖 |
| --- | --- | --- |
| BSA | [cuDNN Frontend v1.29.0](https://github.com/NVIDIA/cudnn-frontend/tree/91dbf3e976a161c1a6833198c708a449330dc3ce) | CUTLASS 4.6.2、TVM FFI 0.1.12、cuDNN 9.15.1.9 |
| causal | [FlashAttention magi_backend](https://github.com/jiayus-nvidia/flash-attention/tree/55221a93a8fc415a721502ed68643983dbf67862) | CUTLASS 4.4.2、TVM FFI 0.1.8.post0、QuACK 0.4.1 |

安装脚本保留已有源码目录。修改内核后，在对应环境重跑安装脚本即可重新构建。

## 一键测试

在 `BSA_ut` 目录中，先查询支持的频率，再运行（2032 MHz 为示例）：

```bash
nvidia-smi -i 0 -q -d SUPPORTED_CLOCKS
GPU=0 bash run_all.sh 2032
```

脚本锁定 GPU 频率，自动选择两个环境，依次测试三个用例。默认预热 5 次、采样 10 次，
每次采样前清 L2；任一测试失败即停止。结束或收到 INT/TERM 后恢复自动频率，
不恢复之前的自定义锁频设置。锁频可能需要管理员权限。

可调整采样次数或 Python 路径：

```bash
GPU=0 WARMUP=5 RUNS=20 bash run_all.sh 2032
BSA_PYTHON=/path/to/bsa/bin/python CAUSAL_PYTHON=/path/to/causal/bin/python \
  GPU=0 bash run_all.sh 2032
```

日志保存在 `results/locked_<时间>_<进程号>/`，可用 `LOG_DIR` 指定新的目录。
三个用例各有一个 `.log`；`clocks.csv` 每 500 ms 记录实际频率、功耗和温度；
其余日志记录配置、GPU 状态及锁频操作。比较结果前检查 `clocks.csv` 是否存在降频，
并保持显存频率、功耗限制及 GPU 独占条件一致。
强制终止或机器重启无法触发清理，必要时执行 `nvidia-smi -i 0 -rgc`。

单项测试可直接选择环境；`--seqlen` 仅适用于两个 causal 用例：

```bash
.venv/bsa/bin/python benchmark.py --case bsa-causal --seqlen 4096
.venv/bsa/bin/python benchmark.py --case production
.venv/causal/bin/python benchmark.py --case causal
```

`bsa-causal` backward 默认使用精确的 block-causal 专用路径。它把
`floor(k / 64) <= floor(q / 64)` 转换为缓存的 packed-mask plan，并使用
SM100/SM103 的每 CTA 128×128、协作式 2-CTA backward kernel。通用 blk64 路径仍可用于
A/B 对照：

```bash
# 先在小 shape 上同时运行两条路径并核对 dQ/dK/dV
.venv/bsa/bin/python -u benchmark.py --case bsa-causal --seqlen 4096 \
  --warmup 2 --runs 10 --bsa-causal-bwd-backend flex --verify-fastpath

# PDF shape 的基线与优化后结果（direct 命令假设 GPU 已提前锁频）
.venv/bsa/bin/python -u benchmark.py --case bsa-causal --seqlen 255424 \
  --warmup 5 --runs 10 --clock-mhz 2032 --bsa-causal-bwd-backend blk64
.venv/bsa/bin/python -u benchmark.py --case bsa-causal --seqlen 255424 \
  --warmup 5 --runs 10 --clock-mhz 2032 --bsa-causal-bwd-backend flex
```

首次调用会构建 mask plan 并 JIT 编译 kernel；预热会把这些一次性成本排除在
采样之外。锁频的一键测试可用
`BSA_CAUSAL_BWD_BACKEND=blk64|flex GPU=0 bash run_all.sh 2032`。

## 输出与计时口径

| 输出列 | 含义 |
| --- | --- |
| `fwd_ms` / `bwd_ms` | 前向 / 反向耗时中位数，毫秒 |
| `fwd_TFLOP/s` / `bwd_TFLOP/s` | 有效矩阵乘吞吐，反向包含 QK 重算 |
| `fwd_MFU_%` / `bwd_MFU_recomp_%` | 吞吐除以单 GPU dense BF16 峰值，百分比 |

令 `P` 为所有 batch/head 的有效可见 Q/K token 对数，`D=128`：
前向 FLOPs 为 `4DP`，反向为 `10DP`（包含 QK 重算）。
BSA 根据实际 active block 统计有效 token 对，排除 padding；标准 causal 使用
`P = B × H × S × (S+1) / 2`。

GB200 锁频峰值模型为 `SM 数 × 8192 × MHz / 10^6` TFLOP/s，SM 数由运行时读取。
152 SM、2032 MHz 对应 **2530.214 TFLOP/s**。单项脚本的 `--clock-mhz` 只用于折算，
实际锁频由 `run_all.sh` 执行；也可用 `--peak-tflops` 指定峰值，两参数互斥。
未提供峰值时，利用率列显示 `—`。

计时使用 CUDA Event，排除输入、mask/Top-K 构造及清 L2；反向复用前向 O/LSE，
包含 BSA backward 内部 CSR 构造。利用率分子仅计有效矩阵乘，忽略 softmax 和 tile 对齐开销，
不代表完整训练 MFU 或硬件活跃周期。若需排除 QK 重算，将反向吞吐和利用率乘以 `0.8`。

## 生产 mask

`production.json` 固定四个互相隔离的样本，共 227519 个有效 token，padding 后为 255424。
`masks.py` 构造可见关系：text 可见本样本全部 token；target video 可见 text，
并从 target/reference video 候选块中选 10% Top-K；reference video 可见 text 并从自身选 Top-K；
audio 可见 text 和自身 audio。10% 是视频候选块比例，不是整个 attention 的密度。

视频按 `1×8×8` cube 排列，每块有效 token 前置。Top-K 按有效 token 的 Q/K block mean 打分，
同分优先较小块编号。输入为合成数据，Q/K/V seed=0、dO seed=1，输入 padding 和 padded query 的 dO 清零。

## Production split backward

The production benchmark defaults to the exact block64 split backward:
K-major CuTe computes dK/dV, while Q-major Triton accumulates dQ once per
Q64 row. Compare it with the fused baseline and validate all gradients before
accepting a performance result:

```bash
# Correctness gate on the full production fixture
.venv/bsa/bin/python -u benchmark.py --case production --warmup 2 --runs 10 \
  --production-bwd-backend split --qmajor-block-n 32 \
  --verify-production-backend

# Locked-clock A/B (run_all defaults to split)
PRODUCTION_BWD_BACKEND=fused GPU=0 bash run_all.sh 2032
PRODUCTION_BWD_BACKEND=split QMAJOR_BLOCK_N=32 GPU=0 bash run_all.sh 2032

# dK/dV bucket and Q-major sub-tile sweep
for bucket in 512 1024 2048 3991; do
  .venv/bsa/bin/python -u benchmark.py --case production --warmup 5 --runs 20 \
    --clock-mhz 2032 --production-bwd-backend split \
    --qmajor-block-n 32 --bucket-size-blocks "$bucket"
done
```

`masks.py` compacts the inactive q2k suffix after Top-K construction. This
does not change the mask; it reduces CSR planning storage and scan work.
The first split invocation JIT-compiles both CuTe and Triton kernels, so keep
warmup enabled. The reported backward TFLOP/s remains the standard effective
five-matmul (10D per visible pair) convention even though split recomputes
QK/dP/dS in its separate dQ kernel.

## Profiling

安装 NVIDIA Nsight Compute 后，可仅采集生产用例的一次反向：

```bash
ncu --nvtx --nvtx-include 'BSA_ut__production__bwd' \
  --set full --target-processes all -o production_bwd \
  .venv/bsa/bin/python benchmark.py --case production --warmup 1 --runs 1
```

NVTX 名称为 `BSA_ut__<case>__fwd` / `BSA_ut__<case>__bwd`，过滤表达式不带末尾 `/`。
结果为 `production_bwd.ncu-rep`；性能基线使用普通运行的耗时，不使用 NCU 采集期间的耗时。

三组用例已在 GB200（L20A）、Torch 2.10 / CUDA 13.1 环境完成前后向性能运行。
`--verify-fastpath` 可对 block-causal 两条 backward 路径做梯度交叉检查；其他
用例仍只测试性能。
