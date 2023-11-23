# -*- coding: utf-8 -*-

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('/opt/qs/aliendao/dataroot/models/moka-ai/m3e-base', device="cuda:1")


def get_emb(query_arr):
    embeddings = model.encode(query_arr).tolist()
    return embeddings


def get_batch_emb(query_arr):
    results = []
    # 分批调用
    sub_arrays = np.array_split(query_arr, 1 if len(query_arr) <= 1000 else (len(query_arr) / 1000 + 1))
    for arr in sub_arrays:
        sub_embs = get_emb(arr.tolist())
        results.extend(sub_embs)
    return results
