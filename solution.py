"""Поиск 50 объявлений для каждого запроса.

Запуск: python solution.py --data data --output answer.csv
Для локальной проверки: python solution.py --data data --output answer.csv --validate
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


SEED = 42
WORD_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)


def clean(value: object) -> str:
    """Одинаково приводим к нижнему регистру запросы и объявления."""
    if not isinstance(value, str):
        return ""
    return value.lower().replace("ё", "е")


def top_indices(indices: np.ndarray, scores: np.ndarray, n: int) -> np.ndarray:
    """Берём лучшие n оценок без полной сортировки большого списка."""
    if len(indices) <= n:
        return indices
    positions = np.argpartition(scores, -n)[-n:]
    return indices[positions]


class Retriever:
    def __init__(self, items: pd.DataFrame, description_limit: int = 1100,
                 params_limit: int = 220):
        self.items = items.reset_index(drop=True)
        self.ids = self.items.item_id.to_numpy()
        self.locations = self.items.item_location_id.to_numpy()
        self.categories = self.items.item_category_id.to_numpy()
        self.microcats = self.items.item_microcat_id.to_numpy()
        self.rating = self.items.item_rating.fillna(0).to_numpy(dtype=np.float32)
        self.reviews = np.log1p(self.items.item_rating_reviews_count.fillna(0).to_numpy(dtype=np.float32))
        self.title = self.items.item_title_raw.map(clean).tolist()
        self.description = self.items.item_description_raw.map(clean).str.slice(0, description_limit).tolist()
        self.params = self.items.item_infm_params_text.map(clean).str.slice(0, params_limit).tolist()
        self.title_words = [set(WORD_RE.findall(s)) for s in self.title]
        self.param_words = [set(WORD_RE.findall(s)) for s in self.params]

        # Заголовок короткий: здесь полезны и слова, и символьные n-граммы.
        # Последние помогают при разных окончаниях слов и опечатках.
        self.vectorizers = [
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300_000,
                            sublinear_tf=True, dtype=np.float32),
            TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3,
                            max_df=0.05, max_features=500_000,
                            sublinear_tf=True, dtype=np.float32),
            TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=400_000,
                            sublinear_tf=True, dtype=np.float32),
        ]
        texts = [self.title, self.title, [d + " " + p for d, p in zip(self.description, self.params)]]
        matrices = [v.fit_transform(t).tocsr() for v, t in zip(self.vectorizers, texts)]
        # Поворачиваем таблицы с весами слов один раз, чтобы не делать
        # эту долгую операцию заново для каждого пакета запросов.
        self.search_matrices = [matrix.T.tocsr() for matrix in matrices]

        # Признаки объявления не зависят от запроса; готовим их один раз.
        self.prices = pd.to_numeric(self.items.item_price, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        self.prices = np.log1p(np.maximum(self.prices, 0))
        self.phone_hidden = self.items.item_is_phone_hidden.to_numpy(dtype=np.float32)
        self.message_forbidden = self.items.item_is_message_forbidden.to_numpy(dtype=np.float32)
        self.latitudes = pd.to_numeric(self.items.item_latitude, errors="coerce").to_numpy(dtype=np.float32)
        self.longitudes = pd.to_numeric(self.items.item_longitude, errors="coerce").to_numpy(dtype=np.float32)
        self.location_centers = {}
        self.location_counts = {}
        self.local_fallback = {}
        for loc, group in self.items.groupby("item_location_id").indices.items():
            group = np.asarray(group)
            self.location_counts[loc] = len(group)
            self.location_centers[loc] = (np.nanmedian(self.latitudes[group]),
                                          np.nanmedian(self.longitudes[group]))
            order = np.argsort(-(self.reviews[group] + self.rating[group] * 0.1))
            self.local_fallback[loc] = group[order[:50]]
        self.global_fallback = np.argsort(-(self.reviews + self.rating * 0.1))[:50]

    def query_vectors(self, queries: pd.DataFrame):
        text = queries.search_query.map(clean).tolist()
        return [v.transform(text).tocsr() for v in self.vectorizers]

    def candidates_and_features(self, queries: pd.DataFrame, per_source: int = 300,
                                behavioral_candidates=None, semantic_candidates=None):
        """Объединяем кандидатов разных видов поиска и считаем признаки."""
        qvecs = self.query_vectors(queries)
        all_features, all_indices, all_offsets = [], [], [0]
        queries = queries.reset_index(drop=True)

        for start in range(0, len(queries), 16):
            if start and start % 512 == 0:
                print(f"  обработано {start}/{len(queries)} запросов", flush=True)
            stop = min(start + 16, len(queries))
            score_blocks = [(q[start:stop] @ x).tocsr() for q, x in zip(qvecs, self.search_matrices)]
            for scores in score_blocks:
                scores.sort_indices()
            for local_row in range(stop - start):
                row = queries.iloc[start + local_row]
                loc = int(row.search_location_id)
                pools = []
                sparse_rows = []
                for scores in score_blocks:
                    sr = scores.getrow(local_row)
                    sparse_rows.append(sr)
                    cols, vals = sr.indices, sr.data
                    pools.append(top_indices(cols, vals, per_source))
                    in_location = self.locations[cols] == loc
                    pools.append(top_indices(cols[in_location], vals[in_location], per_source))

                behavior = (behavioral_candidates[start + local_row]
                            if behavioral_candidates is not None else None)
                if behavior is not None:
                    pools.append(behavior[0])
                if semantic_candidates is not None:
                    pools.append(semantic_candidates[start + local_row][0])

                # Если текстовых совпадений мало, всё равно вернём 50 объявлений.
                candidates = np.unique(np.concatenate(pools)) if pools else np.empty(0, dtype=np.int32)
                if len(candidates) < 50:
                    fallback = self.local_fallback.get(loc, self.global_fallback)
                    candidates = np.unique(np.concatenate([candidates, fallback, self.global_fallback]))

                n_features = 19 if behavior is not None else 17
                features = np.zeros((len(candidates), n_features), dtype=np.float32)
                for channel, sr in enumerate(sparse_rows):
                    # Индексы объявлений отсортированы, поэтому быстро ищем их оценки.
                    positions = np.searchsorted(sr.indices, candidates)
                    valid = positions < len(sr.indices)
                    matches = np.zeros(len(candidates), dtype=bool)
                    matches[valid] = sr.indices[positions[valid]] == candidates[valid]
                    features[matches, channel] = sr.data[positions[matches]]

                features[:, 3] = (self.locations[candidates] == loc)
                features[:, 4] = (self.categories[candidates] == int(row.search_category))
                features[:, 5] = self.rating[candidates]
                features[:, 6] = self.reviews[candidates]
                features[:, 7] = self.prices[candidates]
                features[:, 8] = self.phone_hidden[candidates]
                features[:, 9] = self.message_forbidden[candidates]

                words = set(WORD_RE.findall(clean(row.search_query)))
                if words:
                    for j, item_pos in enumerate(candidates):
                        features[j, 10] = len(words & self.title_words[item_pos]) / len(words)
                        features[j, 11] = float(clean(row.search_query) in self.title[item_pos])
                search_filter = clean(row.search_infm_params_text)
                if search_filter:
                    # Фильтры в данных записаны текстом, поэтому сравниваем их слова.
                    filter_words = set(WORD_RE.findall(search_filter)) - {"вид", "тип", "услуги", "услуг"}
                    if filter_words:
                        for j, item_pos in enumerate(candidates):
                            features[j, 12] = len(filter_words & self.param_words[item_pos]) / len(filter_words)
                features[:, 13] = np.log1p(self.location_counts.get(loc, 0))
                features[:, 14] = len(words)
                features[:, 15] = int(row.search_is_delivery_search)
                if loc in self.location_centers:
                    center_lat, center_lon = self.location_centers[loc]
                    lat_diff = np.radians(self.latitudes[candidates] - center_lat)
                    lon_diff = np.radians(self.longitudes[candidates] - center_lon)
                    a = (np.sin(lat_diff / 2) ** 2 +
                         np.cos(np.radians(center_lat)) * np.cos(np.radians(self.latitudes[candidates])) *
                         np.sin(lon_diff / 2) ** 2)
                    features[:, 16] = np.nan_to_num(12742 * np.arcsin(np.sqrt(np.clip(a, 0, 1))), nan=1000)
                else:
                    features[:, 16] = 1000

                if behavior is not None and len(behavior[0]):
                    # Оценка из истории кликов и место в её выдаче помогают
                    # модели сравнить новые кандидаты с текстовыми.
                    behavior_pos, behavior_scores = behavior
                    rows = np.searchsorted(candidates, behavior_pos)
                    features[rows, 17] = behavior_scores / max(behavior_scores[0], 1e-6)
                    features[rows, 18] = 1 / np.log2(np.arange(len(rows)) + 2)

                all_features.append(features)
                all_indices.append(candidates)
                all_offsets.append(all_offsets[-1] + len(candidates))

        return np.vstack(all_features), np.concatenate(all_indices), np.array(all_offsets)


def make_validation(train: pd.DataFrame, item_ids: set[str]):
    """Проверяем на известных кликах, чьи объявления есть в корпусе."""
    known = train[train.item_id.isin(item_ids)].copy()
    columns = ["search_query", "search_location_id", "search_is_delivery_search",
               "search_infm_params_text", "search_category"]
    known["context"] = known[columns].astype(str).agg("\x1f".join, axis=1)
    groups = known.groupby("context", sort=True)
    contexts = np.array(list(groups.indices))
    rng = np.random.default_rng(SEED)
    rng.shuffle(contexts)
    # Одинаковые запросы целиком относятся либо к обучению, либо к проверке.
    valid_keys = contexts[:750]
    train_keys = contexts[750:3750]
    rows = known.drop_duplicates("context").set_index("context")
    positives = known.groupby("context").item_id.agg(set).to_dict()
    return (rows.loc[train_keys, columns].reset_index(drop=True),
            [positives[k] for k in train_keys],
            rows.loc[valid_keys, columns].reset_index(drop=True),
            [positives[k] for k in valid_keys])


def labels_for_candidates(indices, offsets, positives, item_ids):
    labels = np.zeros(len(indices), dtype=np.int8)
    for q, target in enumerate(positives):
        begin, end = offsets[q:q + 2]
        labels[begin:end] = np.isin(item_ids[indices[begin:end]], list(target))
    return labels


def recall_at_50(scores, indices, offsets, positives, item_ids):
    recalls, coverage = [], []
    for q, target in enumerate(positives):
        begin, end = offsets[q:q + 2]
        order = np.argsort(-scores[begin:end], kind="stable")[:50]
        selected = set(item_ids[indices[begin:end][order]])
        possible = set(item_ids[indices[begin:end]])
        recalls.append(len(target & selected) / len(target))
        coverage.append(len(target & possible) / len(target))
    return float(np.mean(recalls)), float(np.mean(coverage))


def run(data_dir: Path, output: Path, validate: bool):
    print("Читаю файлы...", flush=True)
    items = pd.read_parquet(data_dir / "benchmark_items.parquet")
    queries = pd.read_parquet(data_dir / "benchmark_queries.parquet")
    train = pd.read_parquet(data_dir / "train.parquet")
    print("Строю текстовый индекс...", flush=True)
    retriever = Retriever(items)

    train_queries, train_targets, valid_queries, valid_targets = make_validation(train, set(items.item_id))
    print("Ищу кандидатов для обучения...", flush=True)
    train_x, train_pos, train_offsets = retriever.candidates_and_features(train_queries)
    train_y = labels_for_candidates(train_pos, train_offsets, train_targets, retriever.ids)
    print(f"В обучении {len(train_x):,} пар; найдено {train_y.sum()} положительных.", flush=True)

    print("Ищу кандидатов для проверки...", flush=True)
    valid_x, valid_pos, valid_offsets = retriever.candidates_and_features(valid_queries)
    valid_y = labels_for_candidates(valid_pos, valid_offsets, valid_targets, retriever.ids)
    print(f"В проверке {len(valid_x):,} пар; найдено {valid_y.sum()} положительных.", flush=True)

    # Ranker учится ставить выбранные объявления выше внутри каждого запроса.
    # Это ближе к задаче, чем отдельная классификация каждой пары.
    model = lgb.LGBMRanker(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          min_child_samples=60, objective="lambdarank", eval_at=[50],
                          random_state=SEED, verbosity=-1, n_jobs=8,
                          deterministic=True, force_col_wise=True)
    print("Обучаю модель ранжирования...", flush=True)
    model.fit(train_x, train_y, group=np.diff(train_offsets),
              eval_set=[(valid_x, valid_y)], eval_group=[np.diff(valid_offsets)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    pred = model.predict(valid_x)
    score, coverage = recall_at_50(pred, valid_pos, valid_offsets, valid_targets, retriever.ids)
    print(f"Recall@50={score:.4f}, потолок кандидатов={coverage:.4f}", flush=True)
    if validate:
        id_to_title = dict(zip(retriever.ids, retriever.title))
        missed_by_search, missed_by_model = [], []
        for q, target in enumerate(valid_targets):
            begin, end = valid_offsets[q:q + 2]
            possible = set(retriever.ids[valid_pos[begin:end]])
            top = np.argsort(-pred[begin:end], kind="stable")[:50]
            selected = set(retriever.ids[valid_pos[begin:end][top]])
            if target - possible and len(missed_by_search) < 3:
                item_id = sorted(target - possible)[0]
                missed_by_search.append((valid_queries.iloc[q].search_query, id_to_title[item_id]))
            if (target & possible) - selected and len(missed_by_model) < 3:
                item_id = sorted((target & possible) - selected)[0]
                missed_by_model.append((valid_queries.iloc[q].search_query, id_to_title[item_id]))
        print("Примеры потерь при поиске:", missed_by_search, flush=True)
        print("Примеры потерь при отборе топ-50:", missed_by_model, flush=True)

    print("Ищу объявления для бенчмарка...", flush=True)
    test_x, test_pos, test_offsets = retriever.candidates_and_features(queries)
    test_scores = model.predict(test_x)
    answers = []
    for q in range(len(queries)):
        begin, end = test_offsets[q:q + 2]
        order = np.argsort(-test_scores[begin:end], kind="stable")[:50]
        answers.append(" ".join(retriever.ids[test_pos[begin:end][order]]))
    result = pd.DataFrame({"query_id": queries.query_id, "answer": answers})
    result.to_csv(output, index=False, encoding="utf-8")
    print(f"Сохранено {len(result)} строк в {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    run(args.data, args.output, args.validate)
