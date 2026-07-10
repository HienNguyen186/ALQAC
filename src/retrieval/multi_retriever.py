"""Multi-Retrieval Orchestrator — Tầng 1 của pipeline mới.

Chạy song song 4 retriever, gộp kết quả bằng Reciprocal Rank Fusion (RRF):

  ┌─ BM25 top-200          (sparse lexical)
  ├─ BGE-M3 dense+sparse   (multilingual semantic)
  ├─ Vietnamese Embedding  (PhoBERT, Vietnamese-specific)
  └─ Chunked BM25 top-100  (sliding window, article dài)
        ↓ RRF fusion
  Union candidates ~400-500 (dedup by law_id+aid)

RRF score: sum(1 / (k + rank_i)) với k=60 (standard)
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.retrieval.bm25_retriever import LawRetriever
from src.retrieval.chunked_retriever import ChunkedRetriever
from src.retrieval.vietnamese_retriever import VietnameseRetriever
from src.reranking.dense_reranker import DenseRetriever
from src.utils.io import unique_by_key

LOGGER = logging.getLogger(__name__)

# RRF constant — giá trị chuẩn theo paper (Cormack 2009)
RRF_K = 60


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


class MultiRetriever:
    """Parallel multi-retrieval với RRF fusion.

    Parameters
    ----------
    mode : "mock" | "local"
    corpus_path : str | Path
    bm25_top_k_articles : int   BM25 article-branch pool
    bm25_top_k_laws : int       BM25 law-branch
    bm25_strategy : str         "hybrid" | "article" | "law"
    bgem3_top_k : int           BGE-M3 candidates từ BM25 pool
    vn_top_k : int              Vietnamese embedding candidates
    chunked_top_k : int         Chunked BM25 candidates
    vn_model : str              Vietnamese embedding model
    bgem3_model : str           BGE-M3 model
    batch_size : int
    weight_dense : float        BGE-M3 dense weight
    weight_sparse : float       BGE-M3 sparse weight
    use_parallel : bool         Chạy song song bằng ThreadPoolExecutor
    model_cache_dir : Path | None
    device : str | None
    """

    def __init__(
        self,
        mode: str = "local",
        corpus_path: str | Path = "data/raw/corpus_law_pub.json",
        bm25_top_k_articles: int = 200,
        bm25_top_k_laws: int = 5,
        bm25_strategy: str = "hybrid",
        bgem3_top_k: int = 100,
        vn_top_k: int = 100,
        chunked_top_k: int = 100,
        vn_model: str = "dangvantuan/vietnamese-embedding",
        bgem3_model: str = "BAAI/bge-m3",
        batch_size: int = 32,
        weight_dense: float = 0.6,
        weight_sparse: float = 0.4,
        use_parallel: bool = True,
        model_cache_dir: str | Path | None = None,
        device: str | None = None,
    ):
        self.mode                = mode
        self.bm25_top_k_articles = bm25_top_k_articles
        self.bm25_top_k_laws     = bm25_top_k_laws
        self.bm25_strategy       = bm25_strategy
        self.bgem3_top_k         = bgem3_top_k
        self.vn_top_k            = vn_top_k
        self.chunked_top_k       = chunked_top_k
        self.use_parallel        = use_parallel

        # Tìm model cache dir
        if model_cache_dir:
            _model_cache = Path(model_cache_dir)
        else:
            _here = Path(__file__).resolve()
            _model_cache = next(
                (p / "models" for p in _here.parents if (p / "models").is_dir()),
                None,
            )

        LOGGER.info("[MultiRetriever] Initializing 4 retrievers (mode=%s) ...", mode)

        # ── Retriever 1: BM25 ────────────────────────────────────────
        LOGGER.info("[MultiRetriever] [1/4] BM25 ...")
        self.bm25 = LawRetriever(corpus_path)

        # ── Retriever 2: BGE-M3 (dùng BM25 pool làm input) ───────────
        LOGGER.info("[MultiRetriever] [2/4] BGE-M3 ...")
        self.bgem3 = DenseRetriever(
            mode=mode,
            model_name=bgem3_model,
            batch_size=batch_size,
            weight_dense=weight_dense,
            weight_sparse=weight_sparse,
            model_cache_dir=_model_cache,
            device=device,
        )

        # ── Retriever 3: Vietnamese Embedding ────────────────────────
        LOGGER.info("[MultiRetriever] [3/4] Vietnamese Embedding ...")
        self.vn_retriever = VietnameseRetriever(
            mode=mode,
            model_name=vn_model,
            batch_size=batch_size * 2,  # PhoBERT nhỏ hơn, batch lớn hơn OK
            model_cache_dir=_model_cache,
            device=device,
        )

        # ── Retriever 4: Chunked BM25 ────────────────────────────────
        LOGGER.info("[MultiRetriever] [4/4] Chunked BM25 ...")
        self.chunked = ChunkedRetriever(corpus_path)

        LOGGER.info("[MultiRetriever] All retrievers ready.")

    # ------------------------------------------------------------------
    # Internal: single retriever calls
    # ------------------------------------------------------------------

    def _run_bm25(self, query: str) -> list[dict[str, Any]]:
        """BM25 hybrid pool → top candidates."""
        results = self.bm25.retrieve_candidate_pool(
            query,
            top_k_articles=self.bm25_top_k_articles,
            top_k_laws=self.bm25_top_k_laws,
            strategy=self.bm25_strategy,
        )
        # Gán rank cho RRF
        for rank, r in enumerate(results, 1):
            r["_bm25_rank"] = rank
        return results

    def _run_bgem3(self, query: str, bm25_pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """BGE-M3 dense+sparse trên BM25 pool."""
        results = self.bgem3.retrieve(query, bm25_pool, top_k=self.bgem3_top_k)
        for rank, r in enumerate(results, 1):
            r["_bgem3_rank"] = rank
        return results

    def _run_vn(self, query: str, bm25_pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Vietnamese embedding trên BM25 pool."""
        results = self.vn_retriever.retrieve(query, bm25_pool, top_k=self.vn_top_k)
        for rank, r in enumerate(results, 1):
            r["_vn_rank"] = rank
        return results

    def _run_chunked(self, query: str) -> list[dict[str, Any]]:
        """Chunked BM25 — chạy độc lập trên toàn corpus."""
        results = self.chunked.retrieve(query, top_k=self.chunked_top_k)
        for rank, r in enumerate(results, 1):
            r["_chunked_rank"] = rank
        return results

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_fusion(
        ranked_lists: list[list[dict[str, Any]]],
        rank_fields: list[str],
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion trên nhiều ranked lists.

        Args:
            ranked_lists: Danh sách kết quả từ mỗi retriever
            rank_fields:  Tên field chứa rank trong mỗi list tương ứng

        Returns:
            List articles đã merge và sort theo RRF score, dedup by (law_id, aid)
        """
        # Tích lũy RRF score cho mỗi (law_id, aid)
        rrf_scores: dict[tuple[str, Any], float]       = {}
        art_data:   dict[tuple[str, Any], dict[str, Any]] = {}

        for results, rank_field in zip(ranked_lists, rank_fields):
            for art in results:
                key  = (str(art.get("law_id")), art.get("aid"))
                rank = int(art.get(rank_field, art.get("rank", 9999)))

                rrf_scores[key] = rrf_scores.get(key, 0.0) + _rrf_score(rank)

                # Giữ bản đầy đủ nhất (union of fields)
                if key not in art_data:
                    art_data[key] = dict(art)
                else:
                    art_data[key].update({k: v for k, v in art.items() if k not in art_data[key]})

        # Sort by RRF score
        sorted_keys = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        result = []
        for rank, (key, rrf) in enumerate(sorted_keys, 1):
            entry = dict(art_data[key])
            entry["rrf_score"]       = round(rrf, 6)
            entry["retrieval_score"] = round(rrf, 6)
            entry["rank"]            = rank
            result.append(entry)

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        final_top_k: int = 500,
    ) -> list[dict[str, Any]]:
        """Chạy 4 retrievers song song, fusion bằng RRF, trả về top final_top_k.

        Interface tương thích với LawRetriever.retrieve_candidate_pool()
        để các tầng sau (DenseRetriever, LLMReranker) không cần sửa.

        Returns: list[dict] với fields: law_id, aid, content, rrf_score, rank, ...
        """
        if self.mode == "mock":
            return self._mock_retrieve(query, final_top_k)

        # ── Bước 1: BM25 pool (dùng cho BGE-M3 và VN retriever) ─────
        bm25_pool = self._run_bm25(query)

        # ── Bước 2: Chạy 3 retriever còn lại ─────────────────────────
        if self.use_parallel:
            results_map: dict[str, list] = {}
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(self._run_bgem3,  query, bm25_pool): "bgem3",
                    executor.submit(self._run_vn,     query, bm25_pool): "vn",
                    executor.submit(self._run_chunked, query):           "chunked",
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        results_map[name] = future.result()
                    except Exception as exc:
                        LOGGER.error("[MultiRetriever] %s failed: %s", name, exc)
                        results_map[name] = []

            bgem3_results   = results_map.get("bgem3", [])
            vn_results      = results_map.get("vn", [])
            chunked_results = results_map.get("chunked", [])
        else:
            bgem3_results   = self._run_bgem3(query, bm25_pool)
            vn_results      = self._run_vn(query, bm25_pool)
            chunked_results = self._run_chunked(query)

        LOGGER.info(
            "[MultiRetriever] Candidates: BM25=%d BGE-M3=%d VN=%d Chunked=%d",
            len(bm25_pool), len(bgem3_results), len(vn_results), len(chunked_results),
        )

        # ── Bước 3: RRF Fusion ────────────────────────────────────────
        fused = self._rrf_fusion(
            ranked_lists=[bm25_pool, bgem3_results, vn_results, chunked_results],
            rank_fields=["_bm25_rank", "_bgem3_rank", "_vn_rank", "_chunked_rank"],
        )

        total = len(fused)
        result = fused[:final_top_k]
        LOGGER.info("[MultiRetriever] Fused %d → top %d", total, len(result))
        return result

    # ------------------------------------------------------------------
    # Mock mode
    # ------------------------------------------------------------------

    def _mock_retrieve(self, query: str, final_top_k: int) -> list[dict[str, Any]]:
        """Mock: chỉ dùng BM25 hybrid pool, không cần model."""
        bm25_pool = self.bm25.retrieve_candidate_pool(
            query,
            top_k_articles=self.bm25_top_k_articles,
            top_k_laws=self.bm25_top_k_laws,
            strategy=self.bm25_strategy,
        )
        for rank, r in enumerate(bm25_pool[:final_top_k], 1):
            r["rrf_score"]       = round(_rrf_score(rank), 6)
            r["retrieval_score"] = r["rrf_score"]
            r["rank"]            = rank
        return bm25_pool[:final_top_k]
