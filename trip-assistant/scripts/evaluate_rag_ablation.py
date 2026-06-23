"""RAG 检索消融实验：对比不同检索配置的 top1 / recall@3。

配置：
- bm25_only      仅 BM25 稀疏检索
- vector_only    仅稠密向量检索
- hybrid         BM25 + 向量混合（默认 0.7/0.3）
- hybrid_rerank  hybrid + 重排（--reranker keyword|cross_encoder）

用真实 embedding（--use-configured-embedding），查询走真实分词（terms=None，
不喂手挑关键词），反映真实检索质量。

用法：
    python scripts/evaluate_rag_ablation.py --use-configured-embedding
    python scripts/evaluate_rag_ablation.py --use-configured-embedding --reranker cross_encoder
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.embeddings import EmbeddingManager  # noqa: E402
from rag.local_retriever import LocalMarkdownRetriever  # noqa: E402
from rag.reranker import CrossEncoderReranker, DashScopeReranker, KeywordReranker  # noqa: E402


@dataclass(frozen=True)
class Case:
    case_id: str
    query: str
    expected_document_id: str
    expected_type: str


def get_cases() -> List[Case]:
    """评测集：原始 4 条 + 语义/改写难题（查询与目标无精确关键词重叠）。"""
    return [
        Case("flight-refund", "机票能退吗", "flight_policy", "policy"),
        Case("hotel-cancel", "酒店能取消吗", "hotel_policy", "policy"),
        Case("chengdu-food", "成都有什么好吃的", "chengdu_guide", "guide"),
        Case("sanya-trip", "三亚三天怎么安排", "sanya_guide", "guide"),
        # 语义/改写：查询不含目标文档的精确词，靠语义/字面相似度召回
        Case("refund-money-back", "不想去了能把钱退回来吗", "flight_policy", "policy"),
        Case("hotel-quiet", "想要安静不吵的住处", "hotel_policy", "policy"),
        Case("spicy-food", "想吃麻辣的特色菜", "chengdu_guide", "guide"),
        Case("seaside-relax", "海边发呆躺平放松", "sanya_guide", "guide"),
    ]


def build_retriever(config: str, em: EmbeddingManager) -> LocalMarkdownRetriever:
    if config == "bm25_only":
        return LocalMarkdownRetriever(embedding_manager=em, enable_vector=False)
    if config == "vector_only":
        return LocalMarkdownRetriever(
            embedding_manager=em, enable_vector=True,
            keyword_weight=0.0, vector_weight=1.0, vector_min_score=0.0,
        )
    # hybrid / hybrid_rerank
    return LocalMarkdownRetriever(embedding_manager=em)


def evaluate(retriever, documents, reranker, top_k=3) -> Dict[str, float]:
    cases = get_cases()
    top1_hits = 0
    recall_hits = 0
    details = []
    for case in cases:
        results = retriever.search(query=case.query, documents=documents, top_k=top_k * 2)
        if reranker is not None:
            results = reranker.rerank(case.query, results, top_k=top_k)
        else:
            results = results[:top_k]
        doc_ids = [r.get("document_id") for r in results]
        top1 = results[0].get("document_id") if results else None
        t1 = top1 == case.expected_document_id
        rk = case.expected_document_id in doc_ids
        top1_hits += int(t1)
        recall_hits += int(rk)
        details.append((case.case_id, top1, t1, rk))
    n = len(cases)
    return {
        "top1": top1_hits / n,
        "recall_at_3": recall_hits / n,
        "details": details,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-configured-embedding", action="store_true")
    ap.add_argument("--reranker", choices=["none", "keyword", "cross_encoder", "dashscope"], default="keyword")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    em = EmbeddingManager(api_key=None if args.use_configured_embedding else "")
    documents = _load_documents(LocalMarkdownRetriever(embedding_manager=em))
    print(f"embedding_backend={em.last_backend}  reranker={args.reranker}\n")
    print(f"{'config':>16} {'top1':>6} {'recall@3':>9}")

    reranker = None
    if args.reranker == "keyword":
        reranker = KeywordReranker()
    elif args.reranker == "cross_encoder":
        reranker = CrossEncoderReranker()
    elif args.reranker == "dashscope":
        reranker = DashScopeReranker()

    configs = ["bm25_only", "vector_only", "hybrid"] + (["hybrid_rerank"] if reranker else [])
    summary = {}
    for cfg in configs:
        if cfg == "hybrid_rerank":
            r = build_retriever("hybrid", em)
        else:
            r = build_retriever(cfg, em)
        res = evaluate(r, documents, reranker if cfg == "hybrid_rerank" else None, args.top_k)
        summary[cfg] = {"top1": res["top1"], "recall_at_3": res["recall_at_3"]}
        print(f"{cfg:>16} {res['top1']:>6.2f} {res['recall_at_3']:>9.2f}")

    print("\nper-case (hybrid):")
    r = build_retriever("hybrid", em)
    res = evaluate(r, documents, None, args.top_k)
    for case_id, top1, t1, rk in res["details"]:
        print(f"  {case_id:<22} top1={top1:<16} hit={t1} recall={rk}")
    return 0


def _load_documents(retriever: LocalMarkdownRetriever) -> List[Dict]:
    documents_dir = ROOT / "rag" / "documents"
    documents = []
    documents.extend(retriever.load_documents(documents_dir / "policies", "policy", ROOT))
    documents.extend(retriever.load_documents(documents_dir / "guides", "guide", ROOT))
    return documents


if __name__ == "__main__":
    raise SystemExit(main())
