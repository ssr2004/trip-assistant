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
    chunk_id: str = Field("", description="知识片段唯一标识")
    document_id: str = Field("", description="所属文档标识")
    section: Optional[str] = Field(None, description="所属Markdown章节")
    chunk_index: int = Field(0, description="片段在文档中的序号")
    keyword_score: float = Field(0.0, description="关键词检索分数")
    vector_score: float = Field(0.0, description="向量相似度分数")
    retrieval_strategy: str = Field("keyword", description="命中策略，如keyword、vector或hybrid")


class RetrievalResult(BaseModel):
    """结构化检索结果"""

    query: str = Field("", description="原始查询")
    answer: str = Field("", description="基于检索结果构建的回答")
    sources: List[RetrievedChunk] = Field(default_factory=list, description="引用来源片段")
