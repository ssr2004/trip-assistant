"""
RAG检索结果模型
定义本地知识检索返回的结构化片段
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """结构化检索片段"""

    content: str = Field(..., description="返回给下游使用的片段内容")
    source: str = Field(..., description="知识来源路径或标识")
    type: str = Field(..., description="知识类型，如policy或guide")
    score: float = Field(0.0, description="关键词匹配分数")
    title: Optional[str] = Field(None, description="文档标题")
    matched_terms: List[str] = Field(default_factory=list, description="命中的关键词")
    excerpt: str = Field("", description="可展示的引用片段")


class RetrievalResult(BaseModel):
    """结构化检索结果"""

    query: str = Field("", description="原始查询")
    answer: str = Field("", description="基于检索结果构建的回答")
    sources: List[RetrievedChunk] = Field(default_factory=list, description="引用来源片段")
