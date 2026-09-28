"""Признаки связи слов запроса с подкатегорией услуги.

Для обучающих запросов не берём клики с тем же текстом. Так модель не может
получить подсказку из собственного правильного ответа.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

import numpy as np
import pandas as pd


WORD_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)
STOPWORDS = {"для", "что", "это", "как", "или", "при", "без", "под", "над",
             "услуги", "услуга", "услуг", "нужен", "нужна", "нужно"}


def normalize(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.lower().replace("ё", "е").strip()


def words(value: object) -> tuple[str, ...]:
    return tuple(sorted({w for w in WORD_RE.findall(normalize(value))
                         if len(w) >= 3 and w not in STOPWORDS}))


def text_fold(text: str) -> int:
    number = int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest(), "little")
    return number % 5


class MicrocatPrior:
    """Считает, какие подкатегории часто выбирали по словам запроса."""

    def __init__(self, train: pd.DataFrame, excluded_texts: set[str] | None = None):
        excluded_texts = excluded_texts or set()
        self.token_mc = Counter()
        self.token_total = Counter()
        self.filter_mc = Counter()
        self.filter_total = Counter()
        self.mc_total = Counter()
        self.fold_token_mc = [Counter() for _ in range(5)]
        self.fold_token_total = [Counter() for _ in range(5)]
        self.fold_filter_mc = [Counter() for _ in range(5)]
        self.fold_filter_total = [Counter() for _ in range(5)]
        self.fold_mc_total = [Counter() for _ in range(5)]
        self.total_rows = 0
        self.fold_rows = [0] * 5

        texts = train.search_query.map(normalize).to_numpy()
        filters = train.search_infm_params_text.fillna("").map(normalize).to_numpy()
        microcats = train.item_microcat_id.to_numpy()
        for text, search_filter, mc_value in zip(texts, filters, microcats):
            if text in excluded_texts:
                continue
            mc = int(mc_value)
            fold = text_fold(text)
            self.total_rows += 1
            self.fold_rows[fold] += 1
            self.mc_total[mc] += 1
            self.fold_mc_total[fold][mc] += 1
            for token in words(text):
                self.token_mc[token, mc] += 1
                self.token_total[token] += 1
                self.fold_token_mc[fold][token, mc] += 1
                self.fold_token_total[fold][token] += 1
            if search_filter:
                self.filter_mc[search_filter, mc] += 1
                self.filter_total[search_filter] += 1
                self.fold_filter_mc[fold][search_filter, mc] += 1
                self.fold_filter_total[fold][search_filter] += 1

    def transform(self, query_text: np.ndarray, search_filter: np.ndarray,
                  positions: np.ndarray, offsets: np.ndarray,
                  item_microcats: np.ndarray, leave_fold_out: bool) -> np.ndarray:
        """Возвращает четыре числовых признака для каждой пары."""
        result = np.zeros((len(positions), 4), dtype=np.float32)
        filters_array = search_filter
        for q, raw_text in enumerate(query_text):
            begin, end = offsets[q:q + 2]
            if begin == end:
                continue
            text = normalize(raw_text)
            tokens = words(text)
            filter_text = normalize(filters_array[q])
            fold = text_fold(text) if leave_fold_out else None
            unique_mc, inverse = np.unique(item_microcats[positions[begin:end]],
                                           return_inverse=True)
            values = np.zeros((len(unique_mc), 4), dtype=np.float32)
            for j, mc_value in enumerate(unique_mc):
                mc = int(mc_value)
                probabilities = []
                for token in tokens:
                    count = self.token_mc[token, mc]
                    total = self.token_total[token]
                    if fold is not None:
                        count -= self.fold_token_mc[fold][token, mc]
                        total -= self.fold_token_total[fold][token]
                    probabilities.append(count / (total + 5))
                if probabilities:
                    values[j, 0] = sum(probabilities) / len(probabilities)
                    values[j, 1] = max(probabilities)
                if filter_text:
                    count = self.filter_mc[filter_text, mc]
                    total = self.filter_total[filter_text]
                    if fold is not None:
                        count -= self.fold_filter_mc[fold][filter_text, mc]
                        total -= self.fold_filter_total[fold][filter_text]
                    values[j, 2] = count / (total + 10)
                mc_count = self.mc_total[mc]
                total_rows = self.total_rows
                if fold is not None:
                    mc_count -= self.fold_mc_total[fold][mc]
                    total_rows -= self.fold_rows[fold]
                values[j, 3] = mc_count / max(total_rows, 1)
            result[begin:end] = values[inverse]
        return result
