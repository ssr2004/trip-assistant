"""持久化向量库（可插拔后端）。

解决两个真实问题：
1. ``EmbeddingManager`` 的缓存是内存态，进程重启即失效 → 冷启动重复 embed 全部 chunk。
2. 为 P2 的 FAISS IVF 近似检索提供统一后端接口（exact / FAISS 可切换）。

后端：
- ``InMemoryVectorStore``：精确 cosine，numpy 实现，落盘持久化（.npy + .json）。小语料默认。
- ``FAISSVectorStore``：占位，P2 接入 IVF（含 nlist/nprobe 调参与 crossover 切换）。

设计约定：调用方（检索器）负责把 query 和缺失 chunk 文本合并成一次 embed_batch 调用，
再通过 ``add`` 写入；这样每次检索恒为单次 embedding 调用，同时 chunk 向量被缓存/持久化。
embedding 后端变化（如从降级向量切到真实 Key）时，model_signature 不匹配，
持久化索引自动失效重建，避免维度/语义错配。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """向量库抽象接口。

    所有后端都以 chunk_id 为键存储向量，对外提供 add / search / 持久化能力。
    """

    backend_name = "abstract"

    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = Path(persist_path) if persist_path else None

    def add(self, ids: List[str], embeddings: List[List[float]]) -> None:
        """写入向量并持久化（仅写入尚未存储的 id）。调用方负责批量 embedding。"""
        raise NotImplementedError

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """返回最相似的 (chunk_id, score)，余弦相似度，按分数降序。"""
        raise NotImplementedError

    def get(self, chunk_id: str) -> Optional[List[float]]:
        raise NotImplementedError

    def __contains__(self, chunk_id: str) -> bool:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    """精确 cosine 向量库：内存存储 + 落盘持久化。"""

    backend_name = "inmemory_exact"

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        model_signature: Optional[str] = None,
    ):
        super().__init__(persist_path)
        self.model_signature = model_signature or "unknown"
        self._ids: List[str] = []
        self._id_to_index: Dict[str, int] = {}
        self._matrix: Optional[np.ndarray] = None  # (N, D)，float64 与原 Python 实现一致
        if self.persist_path is not None:
            self.load()

    def add(self, ids, embeddings) -> None:
        new_pairs = [
            (cid, emb)
            for cid, emb in zip(ids, embeddings)
            if cid and cid not in self._id_to_index
        ]
        if not new_pairs:
            return
        self._append([cid for cid, _ in new_pairs], [emb for _, emb in new_pairs])
        if self.persist_path is not None:
            self.save()

    def _append(self, ids: List[str], embeddings: List[List[float]]) -> None:
        if not embeddings:
            return
        vectors = np.asarray(embeddings, dtype=np.float64)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if self._matrix is None:
            self._matrix = vectors
        elif vectors.shape[1] != self._matrix.shape[1]:
            # embedding 后端可能降级（真实调用失败→确定性降级，如 1024→384），
            # 清空旧向量并以新维度重建，避免抛错中断检索（检索层会降级为纯 BM25）。
            logger.warning(
                "embedding 维度变化 %d→%d，向量库清空重建",
                self._matrix.shape[1],
                vectors.shape[1],
            )
            self._ids = []
            self._id_to_index = {}
            self._matrix = vectors
        else:
            self._matrix = np.vstack([self._matrix, vectors])
        for chunk_id in ids:
            self._id_to_index[chunk_id] = len(self._ids)
            self._ids.append(chunk_id)

    def search(self, query_embedding, top_k=10):
        if self._matrix is None or len(self._matrix) == 0:
            return []
        query = np.asarray(query_embedding, dtype=np.float64)
        if query.shape[0] != self._matrix.shape[1]:
            return []
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return []
        dot = self._matrix @ query  # (N,)
        norms = np.linalg.norm(self._matrix, axis=1)
        denom = norms * q_norm
        denom[denom == 0] = 1e-12
        cosine = np.clip(dot / denom, 0.0, None)
        k = min(max(top_k, 1), len(cosine))
        top_idx = np.argpartition(cosine, -k)[-k:]
        top_idx = top_idx[np.argsort(cosine[top_idx])[::-1]]
        return [(self._ids[i], float(cosine[i])) for i in top_idx]

    def get(self, chunk_id):
        idx = self._id_to_index.get(chunk_id)
        if idx is None or self._matrix is None:
            return None
        return self._matrix[idx].tolist()

    def __contains__(self, chunk_id):
        return chunk_id in self._id_to_index

    def __len__(self):
        return len(self._ids)

    def save(self):
        if self.persist_path is None or self._matrix is None:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._persist_npy_path(), self._matrix)
        meta = {
            "ids": self._ids,
            "model_signature": self.model_signature,
            "dim": int(self._matrix.shape[1]),
            "backend": self.backend_name,
        }
        with open(self._persist_json_path(), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def load(self):
        npy = self._persist_npy_path()
        meta_path = self._persist_json_path()
        if not (npy.exists() and meta_path.exists()):
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            stored_signature = meta.get("model_signature")
            if (
                self.model_signature
                and stored_signature
                and stored_signature != self.model_signature
            ):
                # embedding 后端变化，持久化索引失效，重建
                return
            matrix = np.load(npy)
            ids = meta.get("ids", [])
            if matrix.ndim != 2 or len(ids) != matrix.shape[0]:
                return
            self._matrix = matrix.astype(np.float64)
            self._ids = list(ids)
            self._id_to_index = {cid: i for i, cid in enumerate(self._ids)}
        except Exception:
            self._matrix = None
            self._ids = []
            self._id_to_index = {}

    def clear(self):
        self._ids = []
        self._id_to_index = {}
        self._matrix = None

    def _persist_npy_path(self) -> Path:
        # persist_path 视为无扩展名基名，如 data/vector_store/local_index
        return self.persist_path.parent / f"{self.persist_path.name}.npy"

    def _persist_json_path(self) -> Path:
        return self.persist_path.parent / f"{self.persist_path.name}.json"


class FAISSVectorStore(VectorStore):
    """FAISS 近似检索后端（IVF），含 exact/ANN 自动切换。

    - 向量 L2 归一化后用内积（= cosine）检索。
    - 规模低于 ``min_ivf_size`` 时自动退化为 ``IndexFlatIP``（精确，建索引无开销，小语料更快）；
      规模足够时用 ``IndexIVFFlat``（nlist 聚类，nprobe 探测），查询亚线性。
    - nlist 取 ``min(配置值, N/39)``（经验法则 ~sqrt(N)），nprobe 取 ``min(配置值, nlist)``。
    faiss 延迟导入，未安装时实例化仍可用（方法会抛出清晰错误）。
    """

    backend_name = "faiss_ivf"

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        model_signature: Optional[str] = None,
        nlist: int = 128,
        nprobe: int = 8,
        min_ivf_size: int = 512,
    ):
        super().__init__(persist_path)
        self.model_signature = model_signature or "unknown"
        self.nlist = nlist
        self.nprobe = nprobe
        self.min_ivf_size = min_ivf_size
        self._ids: List[str] = []
        self._id_to_index: Dict[str, int] = {}
        self._dimension: Optional[int] = None
        self._raw: Optional[np.ndarray] = None  # (N, D) float32 已归一化，唯一真源
        self._index = None  # faiss 索引，惰性构建
        if self.persist_path is not None:
            self.load()

    def _import_faiss(self):
        try:
            import faiss
            return faiss
        except ImportError as exc:
            raise ImportError("FAISSVectorStore 需要 faiss-cpu：pip install faiss-cpu") from exc

    def add(self, ids, embeddings) -> None:
        new_pairs = [
            (cid, emb)
            for cid, emb in zip(ids, embeddings)
            if cid and cid not in self._id_to_index
        ]
        if not new_pairs:
            return
        faiss = self._import_faiss()
        vecs = np.asarray([emb for _, emb in new_pairs], dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if self._dimension is None:
            self._dimension = int(vecs.shape[1])
        elif vecs.shape[1] != self._dimension:
            raise ValueError(
                f"embedding 维度不一致：{vecs.shape[1]} vs {self._dimension}"
            )
        faiss.normalize_L2(vecs)
        self._raw = vecs if self._raw is None else np.vstack([self._raw, vecs])
        for cid in [cid for cid, _ in new_pairs]:
            self._id_to_index[cid] = len(self._ids)
            self._ids.append(cid)
        self._index = None  # 失效，下次 search 重建
        if self.persist_path is not None:
            self.save()

    def _ensure_index(self):
        if self._index is not None or self._raw is None or len(self._raw) == 0:
            return
        faiss = self._import_faiss()
        vecs = np.ascontiguousarray(self._raw, dtype=np.float32)
        n = len(vecs)
        if n < max(self.min_ivf_size, self.nlist):
            idx = faiss.IndexFlatIP(self._dimension)
            idx.add(vecs)
            self._index = idx
            return
        nlist = max(1, min(self.nlist, n // 39))
        quantizer = faiss.IndexFlatIP(self._dimension)
        idx = faiss.IndexIVFFlat(quantizer, self._dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        idx.train(vecs)
        idx.add(vecs)
        idx.nprobe = max(1, min(self.nprobe, nlist))
        self._index = idx

    @property
    def use_ivf(self) -> bool:
        return self._raw is not None and len(self._raw) >= max(self.min_ivf_size, self.nlist)

    def search(self, query_embedding, top_k=10):
        if self._raw is None or len(self._raw) == 0:
            return []
        self._ensure_index()
        q = np.asarray([query_embedding], dtype=np.float32)
        if q.shape[1] != self._dimension:
            return []
        faiss = self._import_faiss()
        faiss.normalize_L2(q)
        k = min(max(top_k, 1), len(self._ids))
        scores, indices = self._index.search(np.ascontiguousarray(q), k)
        result: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            result.append((self._ids[int(idx)], float(score)))
        return result

    def get(self, chunk_id):
        idx = self._id_to_index.get(chunk_id)
        if idx is None or self._raw is None:
            return None
        return self._raw[idx].tolist()

    def __contains__(self, chunk_id):
        return chunk_id in self._id_to_index

    def __len__(self):
        return len(self._ids)

    def save(self):
        if self.persist_path is None or self._raw is None:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._persist_npy_path(), self._raw)
        meta = {
            "ids": self._ids,
            "model_signature": self.model_signature,
            "dim": int(self._dimension) if self._dimension else 0,
            "backend": self.backend_name,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
        }
        with open(self._persist_json_path(), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def load(self):
        npy = self._persist_npy_path()
        meta_path = self._persist_json_path()
        if not (npy.exists() and meta_path.exists()):
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            stored_signature = meta.get("model_signature")
            if (
                self.model_signature
                and stored_signature
                and stored_signature != self.model_signature
            ):
                return
            raw = np.load(npy)
            ids = meta.get("ids", [])
            if raw.ndim != 2 or len(ids) != raw.shape[0]:
                return
            self._raw = raw.astype(np.float32)
            self._ids = list(ids)
            self._id_to_index = {cid: i for i, cid in enumerate(self._ids)}
            self._dimension = int(raw.shape[1])
        except Exception:
            self._raw = None
            self._ids = []
            self._id_to_index = {}

    def clear(self):
        self._ids = []
        self._id_to_index = {}
        self._raw = None
        self._index = None

    def _persist_npy_path(self) -> Path:
        return self.persist_path.parent / f"{self.persist_path.name}.npy"

    def _persist_json_path(self) -> Path:
        return self.persist_path.parent / f"{self.persist_path.name}.json"
