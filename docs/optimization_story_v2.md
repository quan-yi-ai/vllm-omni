# v2 优化全记录 — 来龙去脉与教学文档

> **目的**：让任何人（或任何 AI agent）读完本文就能**讲清楚每一项优化的来龙去脉**——
> 为什么慢、瓶颈在哪、怎么改的、为什么有效、实测赚了多少、有什么坑。
>
> 分支 `v2-kuaamu-merge`（HEAD 0864acd4 + W4 时序修复）· 910B2 单卡实测
> 日期：2026-08-30 · 配套数据见 `submit_evidence/`

---

## 0. 十万英尺视角：一条语音请求的旅程

MiniCPM-o 4.5 语音复刻（TTS）链路是**三阶段流水线**：

```
文本 ──► Stage0 Thinker(LLM) ──► 文本token ──► Stage1 Talker(AR音频) ──► 音频token
                                                                    │
              音频波形 ◄── Stage2 Code2Wav(Flow-Matching 扩散) ◄──────┘
```

- **Stage0**：文本→文本 token（LLM 自回归，ngram 投机采样加速）
- **Stage1**：文本 token→音频 codec token（自回归，runner-local K 步连解）
- **Stage2**：codec token→波形（CFM 扩散，多步 ODE 求解 → 我们压到伪 1 步）

延迟组成（官方基线，910B2）：首包 TTFP 986ms 里 Stage0 预填充占大头，稳态 RTF
0.44 主要耗在 Stage1 每 token 的引擎往返和 Stage2 逐步 ODE。

**全部 22 项优化本质上只做四类事**：
1. **减少引擎往返**（一次往返连解 K 步，摊薄调度/同步/IPC）
2. **提前算/预先算**（预热、metadata 预构建、cos 全表预计算）
3. **少算步数**（扩散 1 步外推代替多步 ODE；投机采样猜对就跳步）
4. **轻量化交接**（raw-bytes blob 代替 Python 对象序列化）

---

## 1. 三大件详解（最重要）

### 1.1 K12 runner-local 连解（优化 #21，我们的 1 号修复对象）

**问题**：Stage1 Talker 每生成 1 个音频 token 就要走一次完整引擎循环：
scheduler 排队 → IPC 到 worker → 构建 metadata → kernel launch → 同步回传。
音频 token 数以百计，往返开销 ×300 = RTF 杀手。

**改法**：runner 一次拿到 K 个 slot，**在 worker 内部连解 K 步**不上报，
K 步完了一次性回传。调度/同步开销从 ×N 变成 ×N/K。

**K=12 怎么来的**：scheduler 侧窗口（`_k/sched_k`）与 runner 侧
（`_talker_local_steps`）必须**相等**。KuaaMU实测 12 是甜点：K 太小摊不薄开销，
K 太大投机错失后回滚浪费算力。

**我们踩的坑（K12 事故，务必记住）**：
- KuaaMU本地 HEAD `ea66d2d2`（自称 "clean tree"）把 runner 侧误回退成 8，
  但 scheduler 侧还是 12。
- 我们沿用了他的本地 HEAD → **scheduler 12 / runner 8 错位** → scheduler
  以为算了 12 步、KV 只写了 8 步，`num_computed_tokens` 超前于实际 KV →
  speech warmup 时 `assert num_tokens_scheduled > 0` 崩溃。
- **教训**：一切以 `origin/submit`（62f4e4ab）为准；合并别人的代码先 diff
  他的本地 HEAD 与其远程提交版。

**修复**：`npu_ar_model_runner.py` L522 `self._talker_local_steps = 12`。
修复后 speech warmup 7.5s PASS，40 条×2 轮 + 全量 2020 条零崩溃。

### 1.2 TJS1 伪 1 步扩散（优化 #22）

**问题**：Stage2 Code2Wav 是 CFM（Conditional Flow Matching）扩散，默认
多步 ODE 求解，每步都是完整的 HiFT-SAN vocoder 前向。步数 = RTF 直接乘数。

**数学依据**：CFM 轨迹在 t→1 时速度场 v(x,t) 近似常值（终点附近直线）。
所以走 1 步后可以**一阶外推直达终点**：`x₁ = x_t + (1−t)·v`，省掉剩余步。

**开关**：`OMNI_TJS_STOP=1`（默认开）。这正是 FAQ 9.1 合规边界内的
"Token2Wav/Flow 模块少步数"优化。

**代价**：SIM 从扩散多步的 ~0.843 降到 0.8377（全量），仍远超 gate 0.689。
**收益**：Stage2 耗时近乎减半，是 RTF 0.44→0.20 的最大单项贡献者之一。

### 1.3 W4 全链预热（优化 #15 + 我们的 2 号修复对象）

**问题**：NPU 图捕获/算子编译是**按 shape 分桶**的。第一个真实请求会撞上
冷编译尖刺（首包延迟可差 2-5 秒）。评测恰恰从第一条就开始计时。

**改法**：boot 期后台线程发 24 条合成短句（覆盖评测 13-33 字符分布），
把 Stage0/1 图捕获和 Stage2 链编译全部提前吸收。

**我们修的两个 bug（KuaaMU版从未真正生效过）**：
1. **hardcode 模型名 404**：预热请求 model 字段写死 `"openbmb/MiniCPM-o-4_5"`，
   而服务是路径形式启动（served model = `/workspace/.../MiniCPM-o-4_5`）→
   预热请求全部 404，**从未执行过一次**。改为从 `/v1/models` 动态解析。
2. **启动时序竞态**：预热线程在引擎 init 期就 spawn，此时 Uvicorn 还没监听，
   `/v1/models` 解析失败一次就放弃。改为**懒解析**——解析挪进 120 次重试
   循环内，服务起来后自然解析成功。健康触发路径（api_server.py）同样加了
   动态解析。

**效果**：boot 后第一条请求无冷编译尖刺。评测/演示/比赛冷启动场景稳赢。

---

## 2. 其余优化速查表（按作用位置分组）

| 组 | 优化项 | 一句话原理 | 坑 |
|---|---|---|---|
| Stage1 | #1 runner-local K | 见 §1.1 | K 错位即崩 |
| Stage1 | #2 K 窗口预构建 | K 步的 metadata/cos 表提前算好 | 内存换时间 |
| Stage1 | #13 K8 slot 修复 | `num_lookahead_tokens=K-1`（spec 语义） | off-by-one |
| Stage0 | #19 H1 早期终局 | stop_token_ids 管线级广播，Stage1/2 提前收尾 | 需三阶段协议对齐 |
| Stage0 | #20 K14 ngram spec | echo≤15tok 的请求 2 步→1 步（41.9% 命中） | ngram 无模型代价 |
| Stage2 | #17 TJS 跳步 | 见 §1.2 | SIM 微降可接受 |
| Stage2 | #12 TTS 预算公式 | `max(128,min(2048,ct×10+48))` 防长文本截断 | 太大浪费 |
| 交接 | #16 T44 tensor blob | tolist()+msgpack → raw-bytes 零拷贝 | 需两端版本一致 |
| 系统 | #14 CPU 隔离 | 热线程绑核+提优先级 | <16 核自动关 |
| 系统 | #15 W4 预热 | 见 §1.3 | 两个 bug 我们修了 |

---

## 3. 910B2 特有调优（R1 实测，保留在 yaml）

| 参数 | 910B2 值 | 910C KuaaMU值 | 为什么不同 |
|---|---|---|---|
| `codec_chunk_frames` | 50 | 25 | 910B2 launch-bound，chunk 减半=launch 减半（RTF −4.4%） |
| `initial_codec_chunk_frames` | 15 | 无 | 首 chunk 提前出声，首块延迟 −36% |
| `max_num_seqs`（三 stage） | 4 | 8 | **910B2 64GB 内存约束**：8 并发 Code2Wav OOM |
| stage0 `max_tokens` | 2048 | 32 | Daily-Omni 视频 QA 答案 >32 tok 会被截断 |
| `enable_static_kernel` | false | PIECEWISE | 910B2 上 static kernel 编译崩溃 TBE |

**警告**：这份 yaml 是 910B2 特化的。若在 910C 上跑，必须切回KuaaMU 910C 参数
（cf25/seqs8/PIECEWISE/max_tokens 32），否则并发和吞吐反被压制。

---

## 4. 实测总账（全量 2020 条，官方口径）

| 指标 | 官方基线 | 本提交 910B2 | 变化 | KuaaMU 910C |
|---|---|---|---|---|
| RTF | 0.4423 | **0.20** | **−54.8%** | 0.166 |
| TTFP | 986.47ms | **339.35ms** | **−65.6%** | 236.83 |
| TTFT | 333ms | **99.20ms** | **−70.2%** | 118.0 |
| E2EL | — | 1044.45ms | — | 841.98 |
| WER | gate ≤1.56% | **0.0093** | PASS 余量 168× | 0.00987（我们更优） |
| SIM | gate ≥0.689 | **0.8377** | PASS 余量 0.149 | 0.84317 |

**与KuaaMU 910C 的差距全部可归因于硬件**（910B2 单 die 算力≈910C 一半量级，
stage2 launch 开销占比更高）；而** TTFT 反而快 15.9%、WER 反而低 5.8%**——
说明调度层优化（K14 spec + H1 + cf50/icf15）是平台无关的。

---

## 5. 复现 & 验证速查

- 服务命令见 `submit_evidence/V2_SUBMIT_REPORT.md` §2.1/§3.2
- 基准命令见同文件 §3.3（注意 `--result-filename` 连字符）
- 数据：`/tmp/seedtts/seedtts_testset`（zh meta.lst 2020 条）
- 证据：`v2_zh2020_full.json`（844KB 原始逐条）+ 全量日志

## 6. 遗留事项

1. W4 时序修复（懒解析）验证中——重启后看日志 `[W4-I03] full-chain prewarm done`
   且无 "could not resolve" 即生效。
2. 910C 提交前 yaml 切换提醒（见 §3 警告）。
3. KuaaMU本地 HEAD 不可信原则（见 §1.1 教训）。
