"""BM25 关键词检索（CJK 感知分词）。

替代 LocalMarkdownRetriever 原有的子串关键词匹配：
- 子串匹配无词频/文档频率权重，且对中文按空白分词失效。
- BM25 用 TF-IDF + 文档长度归一化，是稀疏检索的事实标准，也是混合检索（hybrid）的稀疏分支。

分词策略：拉丁文按词，CJK 用"单字 + bigram"，兼顾召回与精度，无需外部分词依赖。
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Set

_LATIN_TOKEN = re.compile(r"[A-Za-z0-9]+")
_CJK_CHAR = re.compile(r"[一-鿿]")


def tokenize(text: str) -> List[str]:
    """拉丁文按词、CJK 用单字 + bigram 分词。"""
    if not text:
        return []
    tokens: List[str] = list(tok.lower() for tok in _LATIN_TOKEN.findall(text))
    cjk_chars = _CJK_CHAR.findall(text)
    for i, ch in enumerate(cjk_chars):
        tokens.append(ch)
        if i + 1 < len(cjk_chars):
            tokens.append(ch + cjk_chars[i + 1])
    return tokens


class BM25Scorer:
    """对一个候选 chunk 集合计算 BM25 分数。

    IDF 用 BM25+ 形式 ``log(1 + (N - df + 0.5)/(df + 0.5))`` 保证非负。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score(self, query_tokens: List[str], docs_tokens: List[List[str]]) -> List[float]:
        n = len(docs_tokens)
        if n == 0:
            return []

        df: Dict[str, int] = {}
        doc_len: List[int] = []
        for toks in docs_tokens:
            doc_len.append(len(toks))
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        avgdl = (sum(doc_len) / n) if n else 0.0

        idf: Dict[str, float] = {}
        for t, d in df.items():
            idf[t] = math.log(1 + (n - d + 0.5) / (d + 0.5))

        query_terms: Set[str] = set(query_tokens)
        scores: List[float] = []
        for toks in docs_tokens:
            dl = len(toks)
            if dl == 0:
                scores.append(0.0)
                continue
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            denom_norm = 1 - self.b + self.b * (dl / avgdl if avgdl else 0.0)
            s = 0.0
            for t in query_terms:
                f = tf.get(t)
                if not f:
                    continue
                s += idf.get(t, 0.0) * (f * (self.k1 + 1)) / (f + self.k1 * denom_norm)
            scores.append(max(0.0, s))
        return scores
