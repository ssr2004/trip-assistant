"""FAISS IVF vs 精确检索 crossover 基准。

用合成向量（与真实 embedding 同维度 1024）在不同规模下对比：
- 精确（InMemoryVectorStore，O(N) 暴力 cosine 扫描）
- 近似（FAISSVectorStore，IVF 亚线性）

找出 ANN 查询延迟低于精确、且 recall@10 仍达阈值的拐点，
作为 VectorStore exact/ANN 自动切换阈值的依据。

用法：
    python scripts/benchmark_faiss_crossover.py
    python scripts/benchmark_faiss_crossover.py --scales 1000,5000,10000,50000 --dim 1024
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.vector_store import FAISSVectorStore, InMemoryVectorStore


def gen_vectors(n: int, dim: int, seed: int = 42, n_clusters: int = 100) -> np.ndarray:
    """生成有聚类结构的合成向量（模拟真实语义嵌入）。

    纯随机高斯向量对 IVF 是最坏情况（近邻均匀分散在所有簇），
    会不公平地压低 recall。真实 embedding 因语义聚类而存在簇结构，
    故用 k 个中心 + 小噪声生成，更贴近真实分布。
    """
    rng = np.random.default_rng(seed)
    n_clusters = max(1, min(n_clusters, n))
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-12)
    assign = rng.integers(0, n_clusters, n)
    noise = rng.standard_normal((n, dim)).astype(np.float32) * 0.12
    vecs = centers[assign] + noise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vecs / norms


def time_queries(store, queries, top_k: int):
    times = []
    for q in queries:
        t0 = time.perf_counter()
        store.search(q.tolist(), top_k=top_k)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
    return p50, p95


def topk_ids(store, queries, top_k: int):
    return [
        [cid for cid, _ in store.search(q.tolist(), top_k=top_k)]
        for q in queries
    ]


def recall_at_k(exact_ids, ann_ids):
    if not exact_ids:
        return 1.0
    return len(set(exact_ids) & set(ann_ids)) / len(exact_ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--scales", type=str, default="1000,2000,5000,10000,20000,50000")
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-recall", type=float, default=0.90)
    args = ap.parse_args()
    scales = [int(x) for x in args.scales.split(",") if x.strip()]

    queries = gen_vectors(args.queries, args.dim, seed=7)
    print(f"dim={args.dim}  queries={args.queries}  top_k={args.top_k}\n")
    print(f"{'scale':>8} {'backend':>14} {'build_ms':>10} {'p50_ms':>9} {'p95_ms':>9} {'recall@10':>10} {'ivf':>5}")

    crossover = None
    for n in scales:
        vecs = gen_vectors(n, args.dim)
        ids = [f"c{i}" for i in range(n)]

        exact = InMemoryVectorStore()
        t0 = time.perf_counter()
        exact.add(ids, vecs)
        exact_build = (time.perf_counter() - t0) * 1000
        exact_p50, exact_p95 = time_queries(exact, queries, args.top_k)
        gt = topk_ids(exact, queries, args.top_k)
        print(f"{n:>8} {'exact(cosine)':>14} {exact_build:>10.1f} {exact_p50:>9.3f} {exact_p95:>9.3f} {1.0:>10.2f} {'-':>5}")

        nlist = min(128, max(1, n // 39))
        faiss_store = FAISSVectorStore(nlist=nlist, nprobe=8, min_ivf_size=512)
        t0 = time.perf_counter()
        faiss_store.add(ids, vecs)
        faiss_build = (time.perf_counter() - t0) * 1000
        faiss_store._ensure_index()
        ivf_active = faiss_store.use_ivf
        max_nprobe = getattr(faiss_store._index, "nlist", 1) or 1

        chosen = None
        for nprobe in sorted(set([2, 4, 8, 16, 32, 64, 128, int(max_nprobe)])):
            if ivf_active and nprobe > max_nprobe:
                continue
            if ivf_active:
                faiss_store._index.nprobe = nprobe
            ann = topk_ids(faiss_store, queries, args.top_k)
            rec = sum(recall_at_k(g, a) for g, a in zip(gt, ann)) / len(gt)
            p50, p95 = time_queries(faiss_store, queries, args.top_k)
            chosen = (nprobe, rec, p50, p95)
            if rec >= args.min_recall:
                break
        nprobe, rec, faiss_p50, faiss_p95 = chosen
        print(
            f"{n:>8} {'faiss_ivf':>14} {faiss_build:>10.1f} {faiss_p50:>9.3f} "
            f"{faiss_p95:>9.3f} {rec:>10.2f} {str(ivf_active):>5}  nprobe={nprobe}"
        )

        if crossover is None and faiss_p50 < exact_p50 and rec >= args.min_recall:
            crossover = n

    print(f"\ncrossover（faiss p50 < exact 且 recall@10 >= {args.min_recall}）：{crossover}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
