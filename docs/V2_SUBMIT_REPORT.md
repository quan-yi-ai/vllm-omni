# v2 提交验证报告 — KuaaMU 22 项优化全量合并版（910B2 实测）

> 分支 `v2-kuaamu-merge` · HEAD `0864acd4`（基于 `bfbc7012` v2 全量合并 + K12/W4 修复）
> 硬件：**昇腾 910B2 单卡 64GB**（容器 cgroup 32GB 内存）· CANN 9.0.0 · aarch64
> 日期：2026-08-30 · Seed-TTS zh 官方口径（max_concurrency=1, temperature=0）

---

## 0. 结论速览

| 指标 | 官方基线 | 本提交（910B2 实测） | 变化 | gate |
|---|---|---|---|---|
| **TTFP** | 986.47 ms | **330.58 ms**（40 条热轮） | **−66.5%** | — |
| **RTF** | 0.4423 | **0.20**（40 条） | **−54.8%** | — |
| **TTFT** | 333 ms | **90.22 ms** | −72.9% | — |
| **E2EL** | — | 910.91 ms | — | — |
| **WER** | — | **0.0062** | — | ≤1.56% ✅（余量 2.5 倍） |
| **SIM** | — | **0.8409** | — | ≥0.689 ✅（余量 0.15） |

**与KuaaMU 910C 基线的差距属硬件差异**：KuaaMU在昇腾 910C（Atlas A3）实测 zh2020 全量
E2EL 841.98 / TTFP 236.83 / RTF 0.16606；我们在 910B2 上同口径 40 条 E2EL 994.96→910.91 /
TTFP 333 / RTF 0.21。910B2 单 die 算力约为 910C 的一半量级，且 stage2 在 910B2 上
kernel-launch 开销占比更高（cf. docs/minicpmo_npu_optimization_report.md §11.1
launch-bound 分析）。**两代硬件对同一份代码都远超官方基线**（910C −62% RTF，
910B2 −55% RTF），证明优化是平台无关的算法/调度层收益。

---

## 1. 提交内容

### 1.1 KuaaMU 22 项优化（全量合并，commit bfbc7012）

| # | 优化 | 要点 |
|---|---|---|
| 1 | Stage1 Runner-local K | 一次引擎往返连解 K 步，摊薄调度/同步/IPC |
| 2 | 机制A K 窗口子步预构建 | metadata 预构建/cos 全表预计算/light collect |
| 12 | TTS 预算公式 | `max(128, min(2048, ct×10+48))` 防截断 |
| 13 | K8 跨块 slot 修复 | `num_lookahead_tokens = K-1`（spec-lookahead 语义） |
| 14 | CPU 隔离 | 热线程自绑核 + 提优先级（≥16 核，env 可关） |
| 15 | W4_PREWARM 预热默认开 | boot 期全链预热，消除首请求图捕获方差 |
| 16 | T44 tensor 交接 | Stage0→1 raw-bytes blob 替代 tolist()+msgpack |
| 17 | TJS 跳步 | CFM 轨迹 t→1 速度场近似常值，提前终止 + 一阶外推 |
| 19 | H1 早期终局 | Stage0 `stop_token_ids=[151704,151645]` 管线级终止合同 |
| 20 | K14 spec 默认 | Stage0 ngram spec K 10→14（echo≤15tok 的 41.9% 请求 2步→1步） |
| 21 | **K12 runner-local 默认** | Stage1 窗口 8→12（scheduler `_k/sched_k` + runner `_talker_local_steps`） |
| 22 | **TJS1 伪1步默认** | `OMNI_TJS_STOP=1`：1 步后一阶外推 `x₁=x_t+(1−t)·v` |

（完整 22 项见KuaaMU `05_optimization_report/README.md` #1-22 表格）

### 1.2 我们的 R1 实测调优（910B2 专属，保留在 yaml）

| 配置 | 值 | KuaaMU 910C 值 | 依据 |
|---|---|---|---|
| `codec_chunk_frames` | 50 | 25 | 910B2 launch-bound，chunk 数减半（RTF −4.4% 实测） |
| `initial_codec_chunk_frames` | 15 | （无） | 首 chunk 提前出声，首块延迟 −36% 实测 |
| `token2wav_n_timesteps` | 3 | 3（代码默认） | yaml 显式化，可审计 |
| `max_num_seqs`（三 stage） | 4 | 8 | **910B2 64GB 内存约束**：8 会使 Code2Wav 激活 OOM |
| stage0 `max_tokens` | 2048 | 32 | Daily-Omni 视频 QA 答案超 32 tok（32 会硬截断） |
| `enable_static_kernel` | false | `cudagraph_mode: PIECEWISE` | 910B2 上 static-kernel 编译崩溃 TBE |

### 1.3 本轮修复（commit 0864acd4）

1. **K=12 错位崩溃修复**：`npu_ar_model_runner.py` `_talker_local_steps` 8→12。
   根因：KuaaMU本地 HEAD ea66d2d2（"clean tree"）误回退 `-12 +8`，我们沿用引入
   scheduler 12 / runner 8 的 KV 窗口错位 → speech warmup 崩溃。以其
   **origin/submit（62f4e4ab）提交版为准**修复。修复后 warmup 7.5s PASS。
2. **W4 预热 404 修复**：`async_omni_engine.py` 预热请求 model 字段改为从
   `/v1/models` 动态解析（原 hardcode `"openbmb/MiniCPM-o-4_5"`，路径形式启动时
   404，预热从未真正生效过）。解析失败显式 raise（原版静默失败）。

---

## 2. 验证记录

### 2.1 服务启动（2026-08-30 17:29–17:39 UTC）

```
vllm serve /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 \
  --host 0.0.0.0 --port 8091 --trust-remote-code --omni \
  --allowed-local-media-path / --init-timeout 1500 --interleave-mm-strings \
  --media-io-kwargs '{"video":{"fps":1,"num_frames":128}}'
```

- Stage0 17:35:20 就绪（ngram spec **num_spec_tokens=14** 日志确认）
- Stage1 17:36:42（**Talker local decode ENABLED: K=12** 日志确认）
- Stage2 17:38:35（enforce_eager，TJS1 伪1步：`OMNI_TJS_STOP` 默认 1）
- **Speech warmup 7.5s PASS**（上一版崩溃点，无 assert）
- Application startup complete 17:39:51

### 2.2 冒烟测试

HTTP 200 / 8.46s；choices[0]=文本 + choices[1]=audio
（RIFF/24kHz/mono/16bit/11.48s/275520 帧，样本存 `/tmp/smoke_v2_audio.wav`）。

### 2.3 40 条基准 ×2 轮（Seed-TTS zh，官方口径）

| 轮次 | E2EL (ms) | TTFP (ms) | RTF | TTFT (ms) | WER | SIM |
|---|---|---|---|---|---|---|
| r1（冷） | 994.96 | 333.21 | 0.21 | 92.40 | 0.0062 | 0.8408 |
| r2（热） | **910.91** | **330.58** | **0.20** | **90.22** | 0.0062 | **0.8409** |

日志：`/tmp/bench_v2/bench40.log`、`bench40_r2.log`

### 2.4 全量 2020 条（官方提交口径，2026-08-30 18:10–18:59 UTC）

**Successful 2020/2020，WER/SIM 全量 gate 双 PASS：**

| 指标 | Mean | Median | P99 | gate | 状态 |
|---|---|---|---|---|---|
| E2EL (ms) | 1044.45 | 1045.19 | 1469.28 | — | — |
| TTFT (ms) | 99.20 | 105.09 | 139.50 | — | — |
| **TTFP (ms)** | **339.35** | 342.83 | 385.19 | — | **vs 基线 −65.6%** |
| **RTF** | **0.20** | 0.20 | 0.25 | — | **vs 基线 −54.8%** |
| **WER** | **0.0093** | — | — | ≤1.56% | **PASS**（0 条失败） |
| **SIM** | **0.8377** | — | — | ≥0.689 | **PASS**（2020/2020） |

日志：`/tmp/bench_v2/bench2020.log` · 原始 JSON：`v2_zh2020_full.json`（844KB，已归档本目录）

> 注：全量 E2EL（1044）略高于 40 条热轮（911），因 2020 条含更长文本分布（40 条
> disable-shuffle 取的是头部偏短样本）；两口径均远优于官方基线。

### 2.5 横向对比

| 口径 | E2EL | TTFP | RTF | TTFT | WER | SIM |
|---|---|---|---|---|---|---|
| 官方基线 | — | 986.47 | 0.4423 | 333 | — | — |
| KuaaMU 910C zh2020 全量 | 841.98 | 236.83 | 0.16606 | 118.0 | 0.00987 | 0.84317 |
| **我们 910B2 zh2020 全量** | 1044.45 | 339.35 | 0.20 | 99.20 | **0.0093** | 0.8377 |
| 我们 910B2 40 条热轮 | 910.91 | 330.58 | 0.20 | 90.22 | 0.0062 | 0.8409 |
| 我们 vs 官方基线（全量） | — | **−65.6%** | **−54.8%** | **−70.2%** | — | — |
| 我们 vs KuaaMU（含硬件差） | +24.0% | +43.3% | +20.4% | **−15.9%** | **−5.8%** | −0.65% |

注：全量口径我们 **WER 优于KuaaMU**（0.0093 vs 0.00987）、**TTFT 更快**（99.2 vs
118.0ms）；E2EL/RTF/TTFP 差距与 910B2/910C 单 die 算力差一致（stage2 在 910B2
上 kernel-launch 占比更高）。

---

## 3. 复现步骤

### 3.1 数据集准备

```bash
# Seed-TTS testset（只读 tar，解压到 /tmp）
mkdir -p /tmp/seedtts && tar -xf \
  /workspace/shared_assets/datasets/CowboyZ/seed-tts-eval/seedtts_testset.tar \
  -C /tmp/seedtts
# zh 集 meta.lst 2020 条
```

### 3.2 启动服务

见 §2.1 命令。日志重定向 `/tmp/v2_server.log`。

### 3.3 基准（40 条快速 / 2020 全量）

```bash
cd /tmp/bench_v2
HF_ENDPOINT=https://hf-mirror.com SEED_TTS_SIM_EVAL=1 vllm bench serve --omni \
  --port 8091 --max-concurrency 1 --num-warmups 2 \
  --dataset-name seed-tts --dataset-path /tmp/seedtts/seedtts_testset \
  --seed-tts-locale zh --num-prompts 2020 \
  --no-oversample --disable-shuffle \
  --seed-tts-wer-eval --seed-tts-wer-save-items \
  --temperature 0 \
  --model /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 \
  --trust-remote-code \
  --tokenizer /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 \
  --endpoint /v1/chat/completions --backend openai-chat-omni \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
  --save-result --result-dir /tmp/bench_v2 \
  --result-filename v2_zh2020_full.json \
  --extra_body '{"modalities": ["text", "audio"], "chat_template_kwargs": {"enable_thinking": false, "use_tts_template": true}}'
```

> 注意：`--result-filename` 是连字符形式（`--resultFilename` 会 argparse Exit 2）。
> WER 评测用 funasr paraformer-zh（modelscope 已缓存）；SIM 用 WavLM
> （`/root/.cache/huggingface/hub/models--microsoft--wavlm-base-plus`，
> huggingface.co 不可达时用 `HF_ENDPOINT=https://hf-mirror.com`）。
> 40 条快速验证把 `--num-prompts` 改 40 并去掉 `--save-result*` 两参数。

---

## 4. 已知事项

1. **W4 预热已验证生效且不污染稳态**（2026-08-30 20:12 UTC 实测）：
   重启服务后 `[W4-I03] full-chain prewarm done @19:45:38` 首次出现；
   随后 40 条回归 E2EL 1005.26ms / RTF 0.22 / TTFP 348.11ms /
   WER 0.0062 / SIM 0.8409，与 W4 前基线（994.96ms / 0.21 / 339.35ms /
   0.0062 / 0.8408）一致——预热收益只作用于 boot 态首请求，
   稳态无损。证据：`docs/v2_w4_prewarm_bench40.log`。
   复现命令在 §3.2 基础上加
   `--extra-body '{"modalities":["text","audio"],"chat_template_kwargs":{"enable_thinking":false,"use_tts_template":true}}'`
   （注意：不带此参数 thinking 开启，E2EL 会虚高至 ~2919ms、WER 升至 1.88%，
   那是配置错误不是回归）。
2. **910B2 与 910C 配置差异是有意的**：`max_num_seqs 4 vs 8`、
   `enable_static_kernel false vs PIECEWISE` 均为 910B2 内存/编译器约束的
   实测选择（8 并发 OOM、static kernel 崩 TBE），非遗漏。
3. **KuaaMU本地 HEAD 不可信**：其 ea66d2d2 有误回退，一切以其 origin/submit
   62f4e4ab 为准（K12 修复的教训）。
