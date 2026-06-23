"""LLM-as-Judge RAG 检索质量评测。

用 LLM 对每个检索片段打相关性分（0.0–1.0），作为 RAGAS 式 context precision 的轻量实现，
弥补"精确 document_id 匹配"在语义改写场景下过严的问题（正确文档可能用不同措辞）。

对比 hybrid 与 hybrid + DashScope 重排的 LLM 评判相关性。

用法：
    python scripts/evaluate_rag_llm_judge.py
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.llm import LLMClient, LLMMessage, LLMRequest  # noqa: E402
from core.llm.json_repair import parse_llm_json_object  # noqa: E402
from rag.embeddings import EmbeddingManager  # noqa: E402
from rag.local_retriever import LocalMarkdownRetriever  # noqa: E402
from rag.reranker import BaseReranker, DashScopeReranker  # noqa: E402


JUDGE_SYSTEM = (
    "你是检索质量评判员。给定用户旅行查询和一个检索到的文档片段，"
    "判断该片段对回答/满足查询的相关程度。"
    "只返回一个可解析 JSON 对象：{\"score\": 0.0到1.0的小数, \"reason\": \"不超过20字的中文原因\"}。"
    "不要输出 Markdown 或多余文本。"
)


@dataclass(frozen=True)
class Case:
    case_id: str
    query: str


def get_cases() -> List[Case]:
    return [
        Case("flight-refund", "机票能退吗"),
        Case("hotel-cancel", "酒店能取消吗"),
        Case("chengdu-food", "成都有什么好吃的"),
        Case("sanya-trip", "三亚三天怎么安排"),
        Case("refund-money-back", "不想去了能把钱退回来吗"),
        Case("hotel-quiet", "想要安静不吵的住处"),
        Case("spicy-food", "想吃麻辣的特色菜"),
        Case("seaside-relax", "海边发呆躺平放松"),
    ]


def chunk_text(chunk: Dict) -> str:
    return "\n".join([
        str(chunk.get("title") or ""),
        str(chunk.get("section") or ""),
        str(chunk.get("content") or ""),
    ]).strip()[:600]


async def judge_one(client: LLMClient, query: str, text: str) -> float:
    request = LLMRequest(messages=[
        LLMMessage(role="system", content=JUDGE_SYSTEM),
        LLMMessage(role="user", content=f"查询：{query}\n\n片段：{text}"),
    ])
    response = await client.chat(request)
    try:
        obj = parse_llm_json_object(response.content or "")
        score = float(obj.get("score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


async def evaluate_config(
    client: LLMClient,
    retriever: LocalMarkdownRetriever,
    documents: List[Dict],
    cases: List[Case],
    reranker: Optional[BaseReranker],
    top_k: int,
) -> Dict:
    top1_scores: List[float] = []
    ctx_scores: List[float] = []
    for case in cases:
        results = retriever.search(query=case.query, documents=documents, top_k=top_k * 2)
        if reranker is not None:
            results = reranker.rerank(case.query, results, top_k=top_k)
        else:
            results = results[:top_k]
        if not results:
            continue
        judged = await asyncio.gather(
            *[judge_one(client, case.query, chunk_text(r)) for r in results]
        )
        top1_scores.append(judged[0])
        ctx_scores.append(sum(judged) / len(judged))
    return {
        "top1_relevance": statistics.mean(top1_scores) if top1_scores else 0.0,
        "context_precision": statistics.mean(ctx_scores) if ctx_scores else 0.0,
        "n": len(top1_scores),
    }


def load_documents(retriever: LocalMarkdownRetriever) -> List[Dict]:
    documents_dir = ROOT / "rag" / "documents"
    documents = []
    documents.extend(retriever.load_documents(documents_dir / "policies", "policy", ROOT))
    documents.extend(retriever.load_documents(documents_dir / "guides", "guide", ROOT))
    return documents


async def main_async(args) -> int:
    em = EmbeddingManager()
    retriever = LocalMarkdownRetriever(embedding_manager=em)
    documents = load_documents(retriever)
    client = LLMClient()
    if not client.available:
        print("LLM 不可用（未配置 LLM_API_KEY），无法做 LLM-as-Judge。")
        return 1
    cases = get_cases()

    configs: Dict[str, Optional[BaseReranker]] = {"hybrid": None}
    if args.with_rerank:
        configs["hybrid_dashscope"] = DashScopeReranker()

    print(f"cases={len(cases)} top_k={args.top_k} embedding={em.last_backend or 'n/a'}\n")
    print(f"{'config':>16} {'top1_relevance':>16} {'context_precision':>20}")
    for name, reranker in configs.items():
        res = await evaluate_config(client, retriever, documents, cases, reranker, args.top_k)
        print(
            f"{name:>16} {res['top1_relevance']:>16.2f} {res['context_precision']:>20.2f}"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--with-rerank", action="store_true", help="同时评估 hybrid + DashScope 重排")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
