# 在 6500 条 Claude Code 对话上做混合检索：BM25 + 向量 + rerank 的真实数据

*2026 年 4 月 · `claudegrep` v0.2*

我花了三周时间做了个工具，给自己的 Claude Code 对话历史建索引。结果让我意外的是：**整个项目里最有价值的产物不是代码，是那 51 条手工标注的评测集。**

这篇文章把评测集长什么样、检索效果到底怎么样、踩了哪些坑，按一个工程师能复现的方式写出来。代码全开源（Apache-2.0），评测数据也在仓库里。

> 项目：<https://github.com/lbbstarry/claudegrep> · 安装：`uv tool install claudegrep`

## 起因

每次关掉 Claude Code 终端，那一整个 session 就消失了。`~/.claude/projects/` 下的 JSONL 文件其实都还在，但要从里面找东西只能 `grep`。我经常想得起一句模糊的话——"上次我们怎么决定用 sqlite-vec 而不是 lancedb 来着"——但 grep 显然搜不到这种问题。

需求很简单：**给一段模糊的自然语言，找出三周前的某一轮对话。** 这就是一个标准的稠密检索任务，只不过查询特别短、特别 dev 圈。

## 数据集

127 个 session、6 个项目，跑过分块之后是 **6593 个 chunk**，每个 chunk 平均 1.2k token。

分块策略我对比过三种，最后选了**按"轮"切**（一个 user message + 紧随的 assistant 回复，含 tool_use/tool_result 块直到下一条 user message）：

- **按 message 切**：会把问答拆开，单看一条 assistant message 经常脱离上下文，检索召回的东西没法用。
- **滑动窗口**：索引膨胀 3-5 倍，embedding 成本和延迟都成倍上升，没明显收益。
- **按轮切**：1.2k token 平均长度刚好塞进 bge-m3 的 8k 上下文窗口，不用截断，每个 chunk 自带"问-答"语义对。

## 评测集（这才是核心）

51 条手工标注的 `(query, relevant_chunk_id)` 对，22 个独立目标 chunk。全是从我自己真实历史里采样出来的。

我专门写了个标注 CLI：随机给一个 chunk，让我打一个最可能用来搜到它的查询。**关键约束：查询长度尽量贴近真实场景。**

最后查询长度的中位数是 **5 个字符**。例子：

- `prefab`
- `figma示例`
- `zgame标准`
- `rrf`
- `sqlite vec`

这才是真实开发者搜对话历史的样子——短、模糊、单个项目特定的名词、经常中英混杂。**任何拿一段长自然语言句子做评测的检索系统都在虚报性能**——真实查询根本不长那样。

## 评测结果

WSL2 上的 Ryzen 笔记本，纯 CPU。p95 延迟在热缓存上测。

| 方法 | Recall@10 | MRR | nDCG@10 | p95 ms |
|---|---:|---:|---:|---:|
| BM25 (FTS5) | 0.216 | 0.125 | 0.148 | 2 |
| 向量 (`bge-small-zh-v1.5`, 512 维) | 0.353 | 0.175 | 0.217 | 13 |
| 混合 (RRF, k=60) | 0.392 | 0.175 | 0.228 | 16 |
| **混合 + rerank (`bge-reranker-base`)** | **0.471** | **0.230** | **0.289** | 214 |

复现：`python -m claude_recall.eval.run`。

## 五个超出预期的发现

### 1. BM25 在短 dev 查询上是真的不行

21.6% 的 Recall@10 意味着——五次搜索里有四次目标根本不在前十条词法命中里。开发者对话有大量近重复的文件名、报错碎片、再贴一遍的代码片段，词项重叠度区分不出来谁是真正相关的那一条。

很多 RAG 教程默认 BM25 是个"廉价靠谱的 baseline"。在一般文档库上是这样，在 Claude Code 这种语料上不是。

### 2. 向量检索比 BM25 高 64%

`bge-small-zh` 只有 120MB、512 维，性能依然甩 BM25 一截。最大的提升来自中文查询——这个模型本来就是给中文训练的。

**为什么选 bge-small-zh 而不是 OpenAI text-embedding-3-small？**
- 完全本地、永久免费、隐私零顾虑（这是工具的核心卖点）
- 多语言（我自己代码 + 对话经常中英混杂，多语言模型表现最好）
- 维度小，512 维 vs OpenAI 1536 维，存储开销 1/3
- 拒掉 voyage-3-lite 是因为云端依赖跟"local-first"定位冲突；Pro 付费版会作为可选 backend

### 3. RRF 混合检索只比纯向量高 4 个点

比我预期的少。在长查询场景下，RRF 通常能多 8-10 个点。但短查询里 BM25 列贡献的有效 hit 太少了，融合就没多少东西可融。

**这告诉我：为不同查询长度做自适应权重可能比改 RRF 的 k 值更值得做。**

### 4. Rerank 是单点最大跳跃

+8 点 Recall@10、+6 点 nDCG。cross-encoder 看到完整的查询-文档对（没有 embedding 瓶颈），重排很激进。代价也是真实的：214ms p95 vs 不 rerank 的 16ms。

对交互式 top-10 搜索来说 214ms 完全可以接受；如果是"边输入边搜"那种 UI 你需要先返回 bi-encoder 结果，rerank 异步追加。

### 5. Recall@10 只有 0.471 听起来很差，直到你看查询长什么样

5 个字符无上下文的查询，前 10 条里能命中 50%——这已经从"没法用"跨过了"日常顺手"的门槛。我自己每天都在用。**绝对值放进教科书 benchmark 是难看，但放进真实分布里这就是工作底线。**

## 没起作用的尝试

记录一下白干的活，免得别人重复踩坑：

- **加大 rerank 候选数**（从 30 加到 100）。Recall@10 没变，rerank 变慢。30 个候选已经足够好。
- **embedding 前剥掉代码块。** 每个指标都跌。代码符号查询（`prefab`、`chunks_vec`）就是要把代码留在 embedding 文本里。
- **换成 `bge-base-zh`（768 维）。** 在这个评测集上 Recall@10 没显著提升，embedding 时间翻倍。等 v0.3 直接上 `bge-m3`（1024 维、8k 上下文、原生多语言）。

## 工程上的几点决定（路过的人可以抄）

**单文件 SQLite 干所有事。** FTS5 + sqlite-vec（vec0 虚表）放同一个 `.db` 文件。零运维、随 wheel 一起发布、混合检索就是一次 JOIN。考虑过 Chroma/LanceDB/DuckDB-VSS，在 10 万向量以下规模上 SQLite 全胜——少一个进程、少一份 schema、少一份运维。

**增量索引用 `(mtime, sha256)`。** `files` 表里记每个 JSONL 的 mtime 和 sha256，扫描时遇到没变的直接跳，变了就 delete + re-ingest 那个 session。崩溃安全，幂等。15.7k 条消息全量索引一次 6 分钟，增量 5 秒以内。

**冻结 dataclass 全程不变。** `Message`、`Chunk`、`Session` 都是 `@dataclass(frozen=True)`。`SearchHit` 也是。没有任何就地修改，调试和并发都顺。

**rerank 模型懒加载。** 启动 CLI 时不导入 cross-encoder，只在用户加 `--rerank` 才下载/加载（首次 ~300MB）。普通搜索路径不受影响。

## 下一步

- **bge-m3 + 查询扩展** — 计划中最大的一次提升，v0.3。
- **更大的评测集** — 51 条只是"不是纯噪声"的下限；做 RRF k 值或候选数的微调要 200 条以上才有信号。
- **按项目分桶看指标** — 我怀疑跨项目语义干扰拉低了整体数字，分桶之后单项目内的 Recall 应该好得多。
- **自适应权重** — 短查询给 vector 加权、长查询给 BM25 加权。

## 给做检索系统的同行五点建议

1. **第 2 天就建评测集。** "先把功能跑起来再说"是个陷阱，没数字你不知道改动是好是坏。
2. **用真实查询，别用合成的。** Dev 查询又短又怪。任何 LLM 在你语料上生成的查询读起来都像考试题，不像搜索框。
3. **Recall、MRR、nDCG 一起报。** 各抓一种失败模式：Recall 看"在不在里面"、MRR 看"是不是排在前面"、nDCG 惩罚长尾相关性。
4. **延迟报 p95，不报 p50。** 搜索体验在尾部崩溃。
5. **超过 5k chunk 必须上 rerank。** 比换更好的 embedder 更短的路。

## 仓库 & 评测复现

- 项目：<https://github.com/lbbstarry/claudegrep>
- 评测集：`tests/fixtures/queries.jsonl`
- 评测脚本：`src/claude_recall/eval/run.py`
- 历史结果：`benchmarks/eval_results.md`，每次发版重跑
- 安装：`uv tool install claudegrep` 或 `pipx install claudegrep`

```bash
uv tool install claudegrep
recall index
recall search "rrf 融合" --rerank
```

完全本地、不连云、Apache-2.0。

如果你也用 Claude Code 而且经常想"上次那个事我们怎么聊的"——欢迎试试，提 issue 告诉我哪些查询本来希望能搜到结果但没搜到。这种 bad case 是下一版评测集扩展和 query expansion 的关键输入。

---

*下一篇会写"为什么 sqlite-vec + FTS5 单文件方案在 10 万向量以下完胜 Chroma"，欢迎关注。*
