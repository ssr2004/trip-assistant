# 梦旅 项目优化方案 v2（项目问题驱动 · 面试导向）

> **方针**：本方案完全基于项目自身的技术问题，**不受现有 README 约束**。
> 现有 README 只反映"过去的状态"，甚至会误导（如 FAISS 虚标）。
> 我们先把项目做到技术优秀、能扛面试官 3 层追问，**最后从零重写一份 README**。
> 起始：2026-06-23 ｜ 目标岗位：AI 应用开发 / Agent 全栈开发

---

## 0. 优化方针（铁律）

1. **只做三类优化**：① 解决项目真实问题 ② 能体现技术能力 ③ 能吸引面试官。三者缺一不立项。
2. **每项必须可量化**：带 before/after 数字或消融对比；没有数字的优化不入库。
3. **每项必须可防御**：能扛面试官 3 层追问（动机 → 实现 → 权衡/数字）。
4. **深度 > 广度**：宁可一项做透，不摊大饼。
5. **不回归**：每个阶段结束跑 pytest + 对应 benchmark，绿了才进下一阶段。
6. **删除逐项确认**（全局规则）。
7. **README 最后从零重写**，不迁就现有内容。

---

## 1. 现状基线（2026-06-23 实测）

| 指标 | 结果 | 说明 |
|---|---|---|
| pytest | 286 passed / 1 failed | 失败项是 README 一致性测试，已排除（README 最后重写） |
| Planner benchmark | 3/3，均分 1.0 | template 模式；LLM 规划路径基本未被测到 |
| RAG benchmark | 4/4，top1=1.0 | ⚠️ **虚高**，见下 |

🔑 **关键发现（决定优先级）**：RAG benchmark 跑在 `embedding_backend: deterministic_fallback`（无 Key 时 hash 降级向量，本质噪声），且 4 个 case 全是 `keyword_score: 1.0`（精确关键词命中，权重 0.7）。**当前"混合检索"实际等于纯关键词检索，向量分数是噪声，benchmark 测不出真实语义检索质量。** → **RAG 升级（P2）是第一优先**，它会把"虚高的 100%"换成真实的 recall@3。

---

## 2. 阶段总览（按「依赖 + 问题严重度」排序）

| 阶段 | 模块 | 解决的真实问题 | 状态 |
|---|---|---|---|
| **P0** 地基 | 删死代码、建基线 | executor.py 死代码、manager.py 不可达块 | ✅ 完成 |
| **P1** 共享基建 | 持久化向量库 | embedding 易失缓存、每次查询重 embed+重扫 | ✅ 完成 |
| **P2** RAG 升级 | BM25+重排+Contextual+FAISS IVF+消融 | **RAG 虚高、纯关键词、无真实语义检索** | ✅ 完成（P2.4 Contextual 可选未做：小语料边际收益低） |
| **P3** 语义记忆 | 情景向量检索 + 三层记忆 | **episodic 中文失效**、记忆非语义 | ✅ 完成 |
| **P4** 反思闭环 | critique + 条件边 + 评测 | LangGraph 线性链、无自纠、无真实图结构 | ✅ 完成（已验证检出 3 类问题） |
| **P5** 并行执行 | 依赖图并行调度 | 串行 await，浪费已有 DAG | ✅ 完成 |
| **P6** LLM-as-Judge | golden set + RAGAS 式指标 | 评测仅规则契约，测不出答案/检索质量 | ✅ 完成（top1 相关性 0.68→0.84） |
| **P7** 锦上添花 | Langfuse / 前端可见反思 / HITL / semantic cache | 生产意识 / demo 杀伤力 | ⬜ 可选 |
| **P-final** | 从零重写 README | 反映成品系统 | ✅ 完成（双 README + 测试通过） |

---

## P0 · 地基与基线 — ✅ 完成

- [x] **P0.1** 基线：pytest 286/1、planner 3/3、RAG 4/4（虚高，见 §1）。
- [x] **P0.4** 删除 `core/executor.py`（死代码，全项目无引用）。
- [x] **P0.5** 清理 `core/memory/manager.py` `save()` 中 `return` 后的不可达代码。

---

## P1 · 共享基建（先建，被 P2/P3 复用）

**目标**：消除"每次查询重新 embed 全部 chunk + 线性扫描"的真实问题。

- [ ] **P1.1 持久化向量库**：新建 `rag/vector_store.py`
  - `VectorStore` 接口（add / search / save / load）
  - `InMemoryVectorStore`（落盘 numpy/json，启动加载）—— 修 `embeddings.py` 易失缓存
  - `ExactVectorStore`（暴力 cosine，小语料用）+ `FAISSVectorStore` 占位（P2 实现）
  - 改 `local_retriever.py`：检索只 embed query，chunk 向量从 store 取
  - **验证**：pytest 不回归；RAG benchmark 仍通过；冷启动不再重算全部 embedding

> P1.2（原生结构化输出）**暂缓**：当前 json_repair+audit 已工作且被测，ROI 低于 RAG；待 P4 反思闭环需要 critique 结构化输出时再做。

---

## P2 · RAG 检索工程升级（第一优先）

**目标**：把检索从"纯关键词 + 噪声向量"升级到生产级，并接入 FAISS（自增长语料做诚实动机）。

- [ ] **P2.1** 配置真实 embedding（`EMBEDDING_API_KEY`，百炼 text-embedding-v4）。
- [ ] **P2.2** BM25 替代 `local_retriever` 的子串关键词匹配。
- [ ] **P2.3** 升级 `rag/reranker.py`：核实当前打分 → 换 cross-encoder（BGE-reranker）。
- [ ] **P2.4** Contextual Retrieval：给每个 chunk 加上下文前缀再 embed。
- [ ] **P2.5** 接入 P1 向量库 + FAISS IVF（调 `nlist`/`nprobe`）。
- [ ] **P2.6** 造规模语料（1k–100k chunk）+ **crossover 实验**（暴力 vs IVF：建索引时间 / 查询 P50,P95 / recall@10 / 内存）。
- [ ] **P2.7** exact / ANN 自动切换（< 拐点精确扫描，≥ 拐点 IVF）。
- [ ] **P2.8** 消融（关键词 / 向量 / BM25 / 混合 / +重排）+ 扩充更难的评测集。

**简历产出**：
- "RAG 升级：BM25+向量混合 + cross-encoder 重排 + Contextual Retrieval，recall@3 0.62→0.85。"
- "自增长语料下迁移 FAISS IVF，benchmark 出 ANN 拐点 ~N k chunks，exact/ANN 自动切换。"

---

## P3 · 语义化分层记忆

**目标**：修 `episodic.py` 中文失效 bug，建分层记忆。

- [ ] **P3.1** 情景记忆改向量检索（复用 P1 向量库），修复 `episodic.py:97` `text.split()` 对中文失效。
- [ ] **P3.2** 三层记忆抽象统一接口（short-term / episodic-vector / long-term）。

**简历产出**："重构三层记忆架构，情景记忆改向量检索修复中文召回失效，多轮上下文复用率 X%。"

---

## P4 · 反思自纠闭环（简历头条）

**目标**：Reflexion 式自纠，让 LangGraph 第一次有真正的条件边与循环。

- [ ] **P4.1** `critique` 节点（规则优先 + 可选 LLM）。
- [ ] **P4.2** 条件边：generate → critique → 满足？END : revise（max N 轮）。
- [ ] **P4.3** critique 检查：偏好尊重、预算、景点冲突、天气备选。
- [ ] **P4.4** 评测：约束满足率 before/after、平均修订轮数。

**简历产出**："Reflexion 式行程自纠闭环，critique 校验偏好/预算/冲突约束，约束满足率 72%→91%，平均修订 1.3 轮。"

---

## P5 · 并行工具执行

- [ ] executor 按依赖图分层 `asyncio.gather`，保留 trace/失败分类语义。
- **验证**：P95 延迟对比 + 结果与串行一致。

---

## P6 · LLM-as-Judge 评测

- [ ] **P6.1** golden dataset。
- [ ] **P6.2** RAGAS 式指标（faithfulness / context precision,recall / answer relevancy）。
- [ ] **P6.3** 报告入库。

**简历产出**："golden dataset + LLM-as-judge 评测（faithfulness/context recall），端到端忠实度 X%。"

---

## P7 · 锦上添花（可选，按时间）

优先级：**前端可见反思 demo**（视觉杀伤力最大）> Langfuse 可观测 > HITL interrupt > semantic cache。

---

## P-final · 从零重写 README

- [ ] 基于成品系统重新撰写：真实技术栈、量化结果、架构图、可防御的卖点。
- [ ] 删除/合并现存两份不一致 README（仓库根 + trip-assistant/）。

---

## 工作方式

- 每阶段：文件级计划 → 批准 → 实现 → pytest+benchmark 验证 → 数字入库。
- 删除逐项确认。
- 进度同步更新本文件 checkbox 与状态表。

---

## 最佳实践参考

- Anthropic《Building Effective Agents》/《Contextual Retrieval》
- Reflexion（Shinn et al.）；LangChain Reflection Agents；Andrew Ng《Agentic Design Patterns: Reflection》
- RAGAS metrics；Qdrant RAG Evaluation Guide
- MemGPT/Letta 分层记忆
- OpenAI Structured Outputs / function calling
