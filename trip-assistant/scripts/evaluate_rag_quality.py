"""Run the RAG retrieval quality benchmark suite."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.embeddings import EmbeddingManager  # noqa: E402
from rag.local_retriever import LocalMarkdownRetriever  # noqa: E402


@dataclass(frozen=True)
class RAGQualityCase:
    """单条RAG检索评测用例。"""

    case_id: str
    name: str
    query: str
    terms: List[str]
    expected_document_id: str
    expected_type: str
    top_k: int = 3


def get_default_rag_benchmarks() -> List[RAGQualityCase]:
    """默认RAG检索质量基准集。"""
    return [
        RAGQualityCase(
            case_id="policy-flight-refund",
            name="机票退票政策",
            query="机票能退吗",
            terms=["机票", "退", "退票"],
            expected_document_id="flight_policy",
            expected_type="policy",
        ),
        RAGQualityCase(
            case_id="policy-hotel-cancel",
            name="酒店取消政策",
            query="酒店能取消吗",
            terms=["酒店", "取消"],
            expected_document_id="hotel_policy",
            expected_type="policy",
        ),
        RAGQualityCase(
            case_id="guide-chengdu-food",
            name="成都美食攻略",
            query="成都有什么好吃的",
            terms=["成都", "美食", "火锅"],
            expected_document_id="chengdu_guide",
            expected_type="guide",
        ),
        RAGQualityCase(
            case_id="guide-sanya-trip",
            name="三亚三天攻略",
            query="三亚三天怎么安排",
            terms=["三亚", "三天", "行程"],
            expected_document_id="sanya_guide",
            expected_type="guide",
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate local RAG retrieval quality on curated benchmark cases.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve per case.")
    parser.add_argument("--min-top1-accuracy", type=float, default=0.75, help="Minimum top-1 accuracy.")
    parser.add_argument("--min-recall-at-k", type=float, default=1.0, help="Minimum recall@k.")
    parser.add_argument("--use-configured-embedding", action="store_true", help="Use EMBEDDING_API_KEY from .env.")
    parser.add_argument("--json-compact", action="store_true", help="Print compact JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    embedding_manager = EmbeddingManager(api_key=None if args.use_configured_embedding else "")
    retriever = LocalMarkdownRetriever(embedding_manager=embedding_manager)
    documents = _load_documents(retriever)
    results = [
        evaluate_case(retriever, documents, case, top_k=args.top_k)
        for case in get_default_rag_benchmarks()
    ]
    suite = evaluate_suite(
        results,
        min_top1_accuracy=args.min_top1_accuracy,
        min_recall_at_k=args.min_recall_at_k,
    )
    payload: Dict[str, Any] = {
        "suite": {
            **suite,
            "embedding_backend": embedding_manager.last_backend,
            "use_configured_embedding": args.use_configured_embedding,
        },
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=None if args.json_compact else 2))
    return 0 if suite["passed"] else 2


def evaluate_case(
    retriever: LocalMarkdownRetriever,
    documents: List[Dict],
    case: RAGQualityCase,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """评估单条用例。"""
    limit = top_k or case.top_k
    retrieved = retriever.search(
        query=case.query,
        documents=documents,
        terms=case.terms,
        top_k=limit,
    )
    document_ids = [item.get("document_id") for item in retrieved]
    top1 = retrieved[0] if retrieved else {}
    expected_in_results = case.expected_document_id in document_ids
    top1_hit = top1.get("document_id") == case.expected_document_id
    type_hit = top1.get("type") == case.expected_type if top1 else False
    return {
        "case_id": case.case_id,
        "name": case.name,
        "query": case.query,
        "expected_document_id": case.expected_document_id,
        "expected_type": case.expected_type,
        "top1_hit": top1_hit,
        "recall_at_k_hit": expected_in_results,
        "type_hit": type_hit,
        "retrieved_document_ids": document_ids,
        "top1": _sanitize_chunk(top1),
        "passed": top1_hit and expected_in_results and type_hit,
    }


def evaluate_suite(results: List[Dict[str, Any]], min_top1_accuracy: float, min_recall_at_k: float) -> Dict[str, Any]:
    """汇总评测结果。"""
    total = len(results)
    top1_accuracy = sum(1 for item in results if item["top1_hit"]) / total if total else 0.0
    recall_at_k = sum(1 for item in results if item["recall_at_k_hit"]) / total if total else 0.0
    type_accuracy = sum(1 for item in results if item["type_hit"]) / total if total else 0.0
    return {
        "total": total,
        "top1_accuracy": round(top1_accuracy, 4),
        "recall_at_k": round(recall_at_k, 4),
        "type_accuracy": round(type_accuracy, 4),
        "min_top1_accuracy": min_top1_accuracy,
        "min_recall_at_k": min_recall_at_k,
        "passed": top1_accuracy >= min_top1_accuracy and recall_at_k >= min_recall_at_k,
    }


def _load_documents(retriever: LocalMarkdownRetriever) -> List[Dict]:
    """加载本地政策和攻略文档。"""
    documents_dir = ROOT / "rag" / "documents"
    documents = []
    documents.extend(retriever.load_documents(documents_dir / "policies", "policy", ROOT))
    documents.extend(retriever.load_documents(documents_dir / "guides", "guide", ROOT))
    return documents


def _sanitize_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """只输出评测需要的chunk字段。"""
    return {
        "document_id": chunk.get("document_id"),
        "chunk_id": chunk.get("chunk_id"),
        "title": chunk.get("title"),
        "section": chunk.get("section"),
        "type": chunk.get("type"),
        "score": chunk.get("score"),
        "keyword_score": chunk.get("keyword_score"),
        "vector_score": chunk.get("vector_score"),
        "retrieval_strategy": chunk.get("retrieval_strategy"),
        "source": chunk.get("source"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
