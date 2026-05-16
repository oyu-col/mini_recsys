import gc

import faiss
import numpy as np
import torch
from sklearn.preprocessing import normalize

from kaggle_otto2.util.global_util import GlobalUtil


class SearchUtil:
    @staticmethod
    def ann_search(embeddings, query_embeddings, n_neighbors=50):
        index = faiss.IndexFlatIP(embeddings.shape[1])
        if GlobalUtil.torch_cuda_is_usable() and getattr(faiss, "index_cpu_to_gpu", None):
            try:
                index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
            except Exception:
                index = faiss.IndexFlatIP(embeddings.shape[1])

        embeddings = normalize(embeddings)
        embeddings = embeddings.astype(np.float32)
        query_embeddings = normalize(query_embeddings)
        query_embeddings = query_embeddings.astype(np.float32)

        index.add(embeddings)
        distances, indices = index.search(query_embeddings, n_neighbors)

        del index
        if GlobalUtil.torch_cuda_is_usable():
            torch.cuda.empty_cache()
        gc.collect()

        return distances, indices
