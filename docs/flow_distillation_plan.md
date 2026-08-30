# Token2Wav Flow 少步数蒸馏方案

> 状态：**方案设计**（未实施）。合规依据：官方 FAQ 9.1——允许修改模型权重；
> 9.2——允许投机推理等任意执行侧优化。头部队伍（No.1 RTF 0.1546，基线 35%）
> 几乎必然已走此路线。
>
> 配套文档：`minicpmo_npu_optimization_report.md`（已落地优化的完整记录）。

## 0. 一句话结论

**要做。** 这是当前唯一能突破 RTF 0.52 平台期的路线：推理侧优化（cf50 / nt3 /
TJS / NPUGraph / 融合微优化）已把执行开销压到接近极限，nt=3→2 的负结果证明
**剩余瓶颈不再是步数本身，而是每步 estimator 前向的固有成本**——只有蒸馏能
让"更少的步数"不付精度代价。

## 1. 为什么推理侧已到顶（负结果证据链）

| 实验 | 结果 | 结论 |
|---|---|---|
| n_timesteps 3→2（两轮） | 0.547 → 0.559 / 0.555（统计持平） | 减步数不再提速：固定开销主导 |
| TJS 轨迹跳跃（等效 2 步） | 已含在 0.5238 终验内 | 外推跳步的精度红利已被吃掉 |
| fp16 autocast | 无收益 | launch-bound，非 compute-bound |
| NPUGraph 整图回放 | −2.4%（0.53→0.5174） | launch 开销已压缩 |
| Ascend C 手写融合 kernel | **评估后放弃** | NPUGraph 后剩余为库算子 matmul/conv，CANN 已覆盖，ROI 低 |

**关键洞察**：nt=2 无收益的原因是**音质崩塌风险**（相关性下降）而非速度——
蒸馏恰好解决"少步数 × 高音质"的矛盾。这就是 No.1 队伍 RTF 0.155 的来源：
蒸馏后 1~2 步求解，每步都是"见过 10 步教师轨迹"的学生网络。

## 2. 蒸馏什么、不蒸馏什么

### 2.1 目标：`flow.pt` 的 decoder（DiT estimator），不碰 encoder

```
flow.pt (152.4M 参数) 结构（605 个 key）:
├── input_embedding          (1 key)
├── spk_embed_affine_layer   (2 keys)
├── encoder                  (206 keys, 37.9M)  ← 不蒸馏：UpsampleConformer，跑 1 次/chunk
├── encoder_proj             (2 keys)
└── decoder                  (394 keys, 114.6M) ← 蒸馏目标：DiT-16层，跑 n_timesteps 次/chunk
```

**只蒸馏 decoder.estimator（DiT）**，理由：

1. **频次差 3 倍**：encoder 每 chunk 前向 1 次，DiT 每 chunk 前向
   `n_timesteps`（当前 3，蒸馏后 1~2）次。DiT 是循环内热点。
2. **风险隔离**：encoder 输出 `mu`（条件特征）保持原样，蒸馏学生的输入分布
   不变——教师 encoder 的输出直接喂给学生 DiT，数据管线零改动。
3. **HFiT vocoder（20.8M）不蒸馏**：它是确定性前向（无迭代），蒸馏无意义。

### 2.2 方法：Consistency-style 少步蒸馏（自蒸馏）

**教师** = 原始 DiT，完整 nt=10 余弦调度推理。
**学生** = 同架构 DiT（结构、key 布局与教师完全一致，`load_state_dict
(strict=True)` 直接兼容）。
**数据** = 无需外部标注的自蒸馏对：

```
音频 w（任意语料）
  ├─ speech_tokenizer_v2_25hz.onnx → speech tokens（6561 词表）
  ├─ campplus.onnx                 → spk_embed（192 维）
  └─ mel 谱                        → 目标音频特征（80 维）

教师轨迹: x₀(noise) --[10 步 Euler + CFG 0.7]--> x₁₀(mel)
学生目标: x₀ --[1~2 步]--> x̂，让 x̂ ≈ x₁₀（轨迹级监督）
```

损失函数（两路叠加，先 (a) 后 (a+b)）：

**(a) 轨迹端点蒸馏（主损失，冷启动）**

$$\mathcal{L}_{ep} = \mathbb{E}_{x_0, c}\big[ \| f_\theta(x_0, t_1, c) - x^{*}_{1} \|^2 + \| f_\theta(\hat{x}_1, t_2, c) - x^{*} \|^2 \big]$$

其中 $x^{*}$ 是教师 10 步解的终点，$f_\theta$ 是学生 1~2 步解。直接回归教师
输出 mel。

**(b) 速度场蒸馏（精修，对齐中间行为）**

$$\mathcal{L}_{v} = \mathbb{E}_{t \sim U(0,1)} \big[ \| v_\theta(x_t, t, c) - v_T(x_t, t, c) \|^2 \big]$$

学生在教师轨迹的中间点 $(x_t, t)$ 上对齐教师的速度场 $v_T$（Consistency
Distillation / IMD 的标准做法），保证 CFG 组合
$v = (1+w)v_{cond} - w v_{uncond}$ 下的行为一致。

> 为什么不用 GAN 损失：判别器引入模式坍塌风险 + 训练不稳 + 音质审计困难。
> 赛程内选最稳的回归式蒸馏；GAN 可作为赛后加项。

### 2.3 推理侧改动：零代码，仅换权重 + 调参

交付链路已验证：

```
step_audio2_token2wav.py:141
    self._flow.load_state_dict(torch.load(f"{model_path}/flow.pt"), strict=True)
```

1. 结构不变蒸馏 → 新 `flow.pt` 直接替换模型目录内文件（或通过模型目录
   覆盖 hook），`strict=True` 无感加载；
2. `token2wav_n_timesteps` 从 3 → 1（或 2），`token2wav_jump_steps` → 0
   （1 步求解无需跳步）——纯 yaml/注入层参数，代码不动；
3. NPUGraph 缓存签名不受影响（shape 不变，只有循环次数变）。

### 2.4 数据源（本地已备齐）

| 源 | 规模 | 用途 |
|---|---|---|
| Video-MME 视频音轨（`shared_assets/datasets/lmms-lab/Video-MME`） | 95GB | 主语料：抽 10~20h 干净人声段（VAD 过滤） |
| 模型自带 `assets/*.wav` + `audio_cases/` | 13+ 文件 | 快速冒烟 + 音色回归验证 |
| Seed-TTS testset（本地已有 `/tmp/seedtts`） | 32 样本 | WER gate 自测（≤1.56%） |

数据制造管线（全部本地、无需下载）：

```
ffmpeg 抽音轨 → 16kHz 单声道 → VAD 切 5~15s 段 → 能量/信噪比过滤
→ speech_tokenizer 编码 token + campplus 提 spk_embed + 提 mel 目标
→ (tokens, spk_embed, mel) 三元组缓存为 .pt shard
```

## 3. 训练计划（910B2 单卡，152M 学生 + 152M 教师 = 显存 ~8GB fp16，充裕）

| 阶段 | 内容 | 预计耗时 |
|---|---|---|
| D1 | 数据管线：抽 10h 音频 → 三元组 shard；教师轨迹离线生成（10 步解 + 中间态缓存） | 半天（NPU 批量推理） |
| D1~D2 | 冷启动：端点回归 $\mathcal{L}_{ep}$，~5k step，lr 1e-4 cosine | 数小时 |
| D2~D3 | 精修：+速度场 $\mathcal{L}_{v}$，~10k step | 数小时 |
| D3 | 验收：32 样本 WER ≤ 1.56%？→ 100 样本复测 + Daily-Omni 5 题 spot check | 半天 |
| D3 | 落地：换 flow.pt + `n_timesteps=1`，跑单流 RTF + 4 档文本 | 1 小时 |

**验收红线（不达标则不换权重）**：
- Seed-TTS zh WER（32 样本 seed=1）：≤ 1.56%（与现基线 0.6167% 留 2.5× 余量）
- Daily-Omni spot check：不低于现基线通过数
- 同文本波形与教师版听感无退化（librosa 客观指标 + 抽样人耳）

## 4. 预期收益（保守估计）

当前终验 RTF 0.5238 的构成中 stage2 CFM 循环约占 RTF 的 25~35%（nt=3 已压
到 3 次 DiT 前向/chunk；TJS 后等效 2 次）。蒸馏后等效 1 次前向：

$$RTF_{new} \approx 0.5238 - (2\text{次前向} - 1\text{次前向}) \times \text{单位前向RTF}$$

| 情形 | 估算 |
|---|---|
| 保守（CFM 占 25%，砍 1 次前向） | RTF 0.5238 → **~0.48**（−8%） |
| 中性（CFM 占 35%） | → **~0.46**（−12%） |
| 若同时 nt=2 学生成立（砍 2 次） | → **~0.42**（−20%） |

> 注：蒸馏收益与 TJS 部分重叠（TJS 已等效 2 步）。蒸馏后的净增益 = 回到
> "全精度 2 步" 的质量下再砍到 1 步，即质量恢复 + 再提速的组合。

对比参照：No.1 的 0.1546 说明该路线天花板远高于此，我们的估算只取了最小增量。

## 5. 风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| 少步学生音质退化（电声/嘶哑） | 中 | 两阶段损失先回归后精修；不达标停在 nt=2 学生，仍有收益 |
| 语音内容错误率上升（WER 超门槛） | 低 | 自蒸馏目标即教师输出，分布漂移小；32 样本 gate 先行 |
| Daily-Omni 端到端回归（仅 ~2 题余量） | 中 | **必须**跑 spot check 才换权重；保留旧 flow.pt 秒级回滚 |
| 训练后 NPUGraph 签名失效 | 低 | shape 不变仅循环次数变，无需重捕获；实测确认 |
| 赛程时间不够 | — | 全流程 D1~D3 三天；可先交当前 P0 修复锁定 NPUGraph 收益，蒸馏作为下一轮提交 |

## 6. 决策请求

- [x] 推理侧已到顶的证据链（§1）
- [x] 合规确认（FAQ 9.1/9.2）
- [x] 零代码交付链路验证（strict=True 替换）
- [x] 数据、算力、验收红线齐备
- [ ] **执行**：批准后 D1 开工（数据管线 + 教师轨迹生成）
