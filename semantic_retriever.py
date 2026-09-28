"""Ищет объявления по смысловой близости запроса и заголовка.

Весы модели лежат в папке models/semantic_minilm. Для поиска интернет
не нужен. Векторы объявлений сохраняются рядом с исходными Parquet-файлами.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastembed import TextEmbedding
from numpy.lib.format import open_memmap


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "semantic_minilm"


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def top_positions(scores: np.ndarray, limit: int) -> np.ndarray:
    count = min(limit, len(scores))
    if count == 0:
        return np.empty(0, dtype=np.int32)
    selected = np.argpartition(scores, -count)[-count:]
    # item_id имеет постоянную позицию в корпусе, поэтому равные оценки
    # сортируются одинаково при каждом запуске.
    return selected[np.lexsort((selected, -scores[selected]))].astype(np.int32)


class SemanticTitleIndex:
    def __init__(self, items: pd.DataFrame, data_dir: Path):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Нет локальных весов смысловой модели: {MODEL_PATH}")
        self.encoder = TextEmbedding(
            model_name=MODEL_NAME, specific_model_path=str(MODEL_PATH),
            local_files_only=True, threads=12)
        self.locations = items.item_location_id.to_numpy()
        cache_dir = data_dir / "semantic_experiment"
        cache_dir.mkdir(parents=True, exist_ok=True)
        title_vectors = cache_dir / "title_vectors.npy"
        title_index = cache_dir / "title_index.npy"
        progress = cache_dir / "progress.json"
        if not (title_vectors.exists() and title_index.exists() and progress.exists()):
            self._build_vectors(items.item_title_raw.fillna(""), title_vectors,
                                title_index, progress)
        unique_vectors = np.load(title_vectors, mmap_mode="r")
        index = np.load(title_index, mmap_mode="r")
        ready = json.loads(progress.read_text(encoding="utf-8"))["ready"]
        if len(index) != len(items) or ready != len(unique_vectors):
            raise RuntimeError("Кеш смыслового поиска не совпадает с корпусом")
        self.vectors = np.asarray(unique_vectors[index], dtype=np.float32)

    def _build_vectors(self, titles: pd.Series, title_vectors: Path,
                       title_index: Path, progress: Path) -> None:
        unique_titles, inverse = np.unique(titles.to_numpy(dtype=str),
                                           return_inverse=True)
        np.save(title_index, inverse.astype(np.int32))
        vectors = open_memmap(title_vectors, mode="w+", dtype=np.float32,
                              shape=(len(unique_titles), 384))
        for start in range(0, len(unique_titles), 2048):
            stop = min(start + 2048, len(unique_titles))
            block = np.asarray(list(self.encoder.embed(
                unique_titles[start:stop].tolist(), batch_size=256)), dtype=np.float32)
            vectors[start:stop] = normalize(block)
            vectors.flush()
            progress.write_text(json.dumps({"ready": stop, "total": len(unique_titles)}),
                                encoding="utf-8")
            print(f"Векторы заголовков: {stop}/{len(unique_titles)}", flush=True)

    def search_scored(self, queries: pd.DataFrame,
                      global_limit: int = 300, local_limit: int = 300):
        """Для каждого запроса берёт лучшие объявления по стране и городу."""
        query_vectors = normalize(np.asarray(list(self.encoder.query_embed(
            queries.search_query.fillna("").tolist(), batch_size=256)), dtype=np.float32))
        results = []
        for start in range(0, len(queries), 32):
            stop = min(start + 32, len(queries))
            similarities = query_vectors[start:stop] @ self.vectors.T
            for local_q in range(stop - start):
                q = start + local_q
                row = similarities[local_q]
                global_items = top_positions(row, global_limit)
                same_city = np.flatnonzero(self.locations == queries.search_location_id.iloc[q])
                local_items = same_city[top_positions(row[same_city], local_limit)]
                positions = np.union1d(global_items, local_items).astype(np.int32)
                scores = row[positions]
                order = np.lexsort((positions, -scores))
                results.append((positions[order], scores[order].astype(np.float32)))
        return results, query_vectors

    def scores_for_candidates(self, query_vectors: np.ndarray,
                              positions: np.ndarray, offsets: np.ndarray,
                              retrieved: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Считает смысловое сходство и место в выдаче для всего пула."""
        extra = np.zeros((len(positions), 3), dtype=np.float32)
        for q, vector in enumerate(query_vectors):
            begin, end = offsets[q:q + 2]
            candidates = positions[begin:end]
            extra[begin:end, 0] = self.vectors[candidates] @ vector
            chosen, _ = retrieved[q]
            if len(chosen):
                rows = np.searchsorted(candidates, chosen)
                extra[begin + rows, 1] = 1
                extra[begin + rows, 2] = 1 / np.log2(np.arange(len(rows)) + 2)
        return extra
