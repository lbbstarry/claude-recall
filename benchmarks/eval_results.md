# claude-recall eval (n=51, k=10)

Embedder: `BAAI/bge-small-zh-v1.5` (dim=512)

| method | recall@10 | MRR | nDCG@10 | p95 ms |
|---|---:|---:|---:|---:|
| BM25 | 0.216 | 0.125 | 0.148 | 2.0 |
| Vector (BAAI/bge-small-zh-v1.5) | 0.353 | 0.175 | 0.217 | 13.1 |
| Hybrid (RRF) | 0.392 | 0.175 | 0.228 | 15.6 |
| Hybrid + rerank | 0.471 | 0.230 | 0.289 | 213.9 |
