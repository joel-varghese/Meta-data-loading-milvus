import time
import numpy as np
import torch
from torch.nn import functional as F


def embed_query(encoder, query):
    query_embeddings = torch.tensor(encoder.encode(query))

    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

    norms = np.linalg.norm(query_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5) == True

    query_embeddings = list(map(np.float32, query_embeddings))

    return query_embeddings


def client_assemble_retrieved_context(retrieved_top_k, metadata_fields=[], num_shot_answers=3):
    """
    Assemble context and metadata from top k chunks
    """

    distances = []
    context = []
    context_metadata = []
    i = 1
    for r in retrieved_top_k[0]:
        distances.append(r['distance'])
        if i <= num_shot_answers:
            if len(metadata_fields) > 0:
                metadata = {}
                for field in metadata_fields:
                    metadata[field] = r['entity'][field]
                context_metadata.append(metadata)
            context.append(r['entity']['chunk'])
        i += 1
    
    formatted_results = list(zip(distances, context, context_metadata))

    return formatted_results, context, context_metadata