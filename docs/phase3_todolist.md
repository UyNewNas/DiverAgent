# Phase 3 (类比推理) 任务清单

## 科学问题

方向性探针能否学习"类比方向"（A→B 的关系向量），在未见过的头实体上沿该方向生成多样、合理、新颖的尾实体？即：`Head + transport_k(Head⊕Relation) → Tail_k`

## 与 Phase 2b 的核心差异

| 维度 | Phase 2b (MIT-States) | Phase 3 (类比推理) |
|------|----------------------|---------------------|
| 域 | 图像像素空间 | 概念嵌入空间 |
| 输入 | 图像 (128×128×3) | 概念词向量 (300D) |
| 输出 | 生成图像 | 尾实体嵌入向量 |
| Encoder | CNN from scratch | 预训练词嵌入 (GloVe/fastText) |
| Decoder | Deconv 网络 (128×128) | MLP MLP (300→tail_emb) |
| 评估 | DCI + shape/color retention | DCI + analogy_accuracy + distinct-N |
| 数据 | MIT-States (~60k 图像) | WordNet/ConceptNet 类比对 (数千条) |
| CLIP | 图像-文本对齐 | 不适用 |

## 跨 Phase 遗产（直接继承）

| 遗产 | 来源 | Phase 3 中的形式 |
|------|------|-----------------|
| 平方锚定损失 | Phase 0 | `0.15×(l_div-0.7)² + 0.8×l_pla + 0.5×l_nov` (不变) |
| 逐层注入 | Phase 1 | Decoder 每层 MLP 加 `inj_mlp_i(z)` |
| 方向性探针 | Phase 2a | `z_k = head_emb + transport_k(head⊕rel) + rel_emb` |
| 无 scheduler 训练 | Phase 2a | 固定 lr=1e-3, 无 CosineAnnealingLR |
| NoveltyLoss | Phase 0-2b | threshold 从 0.80 下调 → 0.65 (Phase 2b 教训) |
| 随机基线 + 消融 | Pre-Phase 2 | 评估标配, 消融从 fresh init 开始 |
| 训练种子 | Pre-Phase 2 | `random.seed(42); torch.manual_seed(42)` |
| 数据诚实性核查 | Phase 2b | 所有报告数据必须可从 JSON + 日志回溯 |

---

## E1：搭建项目骨架

| 子任务 | 说明 | 产出 |
|--------|------|------|
| 1.1 创建目录 | `experiments/analogy/checkpoints/`, `results/`, `data/` | 目录结构 |
| 1.2 `config.py` | 超参数：EMBED_DIM=300, HIDDEN_DIM=256, RELATION_DIM=128, K_OUTPUTS=8, BATCH_SIZE=256, λ=(0.15, 0.8, 0.5), NOVELTY_THRESHOLD=0.65, 各阶段 epochs | 配置文件 |
| 1.3 `dataset.py` | 构造 WordNet/ConceptNet 类比对：`(head, relation, tail)` 三元组。按 relation type 划分 train/test，确保 test 中的 (head, tail) 配对在 train 中未出现 | 数据加载器 |
| 1.4 `__init__.py` | 空文件 | — |

**验收**：`python -c "from experiments.analogy.config import *; print(EMBED_DIM)"` 正常；`dataset.py` 能统计 train/test pair 数量和重叠检查

---

## E2：实现 Backbone

| 子任务 | 说明 | 关键约束 |
|--------|------|----------|
| 2.1 词嵌入加载 | 加载 GloVe 300D 或 fastText 预训练向量 → `nn.Embedding.from_pretrained` (freeze) | 冻结，不训练 |
| 2.2 Encoder | `head_emb + rel_emb → MLP(600→HIDDEN_DIM→RELATION_DIM)` 提取关系方向向量 | 2 层 MLP + ReLU |
| 2.3 Relation Head | `nn.Embedding(num_relations, RELATION_DIM)`, 训练时 `MSE(encoder_out, rel_embed[rel_id])` 对齐 | 类似 Phase 2b 的 obj/attr embedding |
| 2.4 Directional Probe | `transport_mlp(head⊕rel → RELATION_DIM×K)`, `z_k = head_emb + transport_k + rel_emb` | 继承 Phase 2a 方向性 |
| 2.5 Decoder | `fc(z_k → HIDDEN_DIM → EMBED_DIM)`, 多层 MLP + 逐层 injection | 每层 MLP 后注入探针信号 |
| 2.6 DivergenceGate | `nn.Linear(RELATION_DIM→64→1→Sigmoid)` | 延后验证 |
| 2.7 CBDP 模型 | 组装以上所有模块 + `compute_divergent_loss` | — |
| 2.8 参数统计 | 打印 `probe_ratio`, 必须 ≤3% | 硬约束 |

**验收**：`model = AnalogyCBDP()` 不报错，`model.parameter_stats()` 输出比例 ≤3%

---

## E3：Stage 1 训练收敛主干

| 子任务 | 说明 |
|--------|------|
| 3.1 `stage1_train_backbone.py` | 加载类比数据 → 训练 Encoder + Decoder (无探针) |
| 3.2 损失 | `MSE(reconstructed_tail, true_tail_emb) + 0.1×rel_alignment` |
| 3.3 优化器 | Adam(lr=1e-3), CosineAnnealingLR (主干可保留 scheduler) |
| 3.4 Epochs | 50-100, 根据 loss 收敛情况 early stop |
| 3.5 日志 | 重定向到 `results/train_stage1.log` |

**验收**：Backbone 能根据 (head, relation) 重建 tail 嵌入（cos_sim 与 true tail > 0.5）；`backbone_best.pt` 存在

---

## E4：Stage 2 训练发散探针

| 子任务 | 说明 |
|--------|------|
| 4.1 `stage2_train_probe.py` | 加载 Backbone → 构建记忆库 → 冻结主干 → 训练探针 |
| 4.2 **无 scheduler** | 固定 lr=1e-3, 20 epochs (Phase 2a 教训) |
| 4.3 记忆库 | `build_memory_bank`: 存训练样本的 tail_emb, max_samples=N, `is_full=True` |
| 4.4 损失 | `0.15×(l_div-0.7)² + 0.8×l_pla + 0.5×l_nov`, threshold=0.65 |
| 4.5 优化器 | Adam(list(probe.params) + list(gate.params), lr=1e-3) |
| 4.6 日志 | 每 epoch 打印 div/pla/nov/total, 重定向到 `results/train_stage2.log` |

**验收**：`cbdp_full.pt` 存在, 训练日志 novelty > 0 (threshold 降低后应不再为 0)

---

## E5：评估

| 子任务 | 说明 |
|--------|------|
| 5.1 DCI (3维) | `(pla × div_quality × novelty)^(1/3)`, τ=0.3, 对 Backbone 和 CBDP 都算 |
| 5.2 cos_sim | K 输出 tail 嵌入之间的成对余弦相似度 |
| 5.3 Plausibility | `1 - min_K MSE(probe_tail_k, true_tail_emb)` |
| 5.4 Novelty | K 输出 vs 记忆库 tail_emb 余弦相似度 (1 - max_sim) |
| 5.5 **类比准确率 (Top-1)** 🆕 | `argmax_k cos_sim(probe_tail_k, true_tail)` 的命中率 |
| 5.6 **类比准确率 (Top-3)** 🆕 | 与 true tail cos_sim 排名前 3 的命中率 |
| 5.7 **Distinct-K** 🆕 | K 个输出 tail_emb 中彼此 cos_sim < 0.7 的比例 (嵌入空间多样性) |
| 5.8 随机基线 | `transport_k = randn × scale ∈ {0.05, 0.1, 0.3, 0.5}`, 取最优 |
| 5.9 消融实验 | λ_div=0 / λ_pla=0 / λ_nov=0 / full, **fresh init** 各训 10 epochs 无 scheduler |

**验收**：类比准确率 > 随机基线；随机基线 DCI < CBDP DCI

---

## E6：可视化 + 定性分析

| 子任务 | 说明 |
|--------|------|
| 6.1 `visualize.py` | 对抽样测试类比，可视化：head → [K 个 tail candidates], 标注与 true tail 余弦相似度 |
| 6.2 最近邻检索 | 对每个 probe_tail_k, 在词汇表中检索 cos_sim 最高的 top-3 词, 输出人类可读的类比结果 |
| 6.3 失败案例分析 | 选取 DCI 最低的 10 个测试类比，分析失败模式（关系误解？head 未见？） |

---

## E7：一键运行

| 子任务 | 说明 |
|--------|------|
| 7.1 `run_all.py` | 串联 E1→E6：数据准备 → Stage 1 → Stage 2 → 评估 → 可视化 |

---

## E8：Git 留痕

| 子任务 | 说明 |
|--------|------|
| 8.1 每步 commit | E1→E7 每个完成一个 commit, 类型 `impl` / `fix` / `result` |
| 8.2 Tag | 关键 commit 打 tag：`phase3-v1-initial` 等 |
| 8.3 `.gitignore` | 排除 `checkpoints/*.pt`, `data/`, `__pycache__/` |

---

## 成功标准

| 指标 | 底线 | 理想 |
|------|:---:|:---:|
| DCI (test unseen pairs) | ≥ 0.35 | ≥ 0.50 |
| CBDP vs Random ΔDCI | > +50% | > +100% |
| cos_sim | 0.15-0.50 | 0.20-0.35 |
| 类比 Top-1 准确率 | ≥ 0.15 | ≥ 0.30 |
| 类比 Top-3 准确率 | ≥ 0.30 | ≥ 0.50 |
| Distinct-K | ≥ 0.30 | ≥ 0.50 |
| Plausibility | ≥ 0.60 | ≥ 0.80 |
| Novelty (训练) | > 0 | > 0.10 |
| 探针占比 | ≤ 3% | ≤ 2% |
| train-test gap | < 0.05 | < 0.02 |

---

## 预估时间

| 步骤 | 时间 |
|------|------|
| E1-E2 代码搭建 | ~3h |
| E3 Backbone 训练 | ~1h (嵌入空间训练极快) |
| E4 探针训练 | ~15min |
| E5 评估+基线+消融 | ~15min |
| E6 可视化+定性 | ~30min |
| **总计** | **~5h** |

---

## 关键风险

| 风险 | 等级 | 缓解措施 |
|------|:---:|------|
| GloVe 嵌入对类比关系表达能力不足 | 🟠 高 | 备选 fastText 或小型 BERT 嵌入 |
| Novelty 再次为 0 (threshold 偏高) | 🟡 中 | threshold 从 0.65 起，按训练日志调至 0.55-0.60 |
| 类比数据量不足 (数千对) | 🟡 中 | 数据增强：同关系的不同 head 互换构造伪类比 |
| 嵌入空间 DCI 缺乏图像直观验证 | 🟢 低 | 用最近邻检索转成人类可读的词，做定性验证 |
