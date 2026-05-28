from collections import defaultdict
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = SentenceTransformer(model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2")

def compute_embedding_from_chunks(chunks: list):
  """
  Compute embeddings for a list of text chunks.

  Args:
    chunks (list of dict): A list of dictionaries, each containing a 'chunk' key with the text to be embedded.
  Returns:
    A list of embeddings corresponding to the input chunks.
  """
  sentences = ["\n".join(x for x in [" > ".join(item.get("header_path", [])), item.get("chunk", "")] if x) for item in chunks]
  embeddings = EMBEDDING_MODEL.encode(inputs = sentences, show_progress_bar = False, convert_to_numpy = True)
  return embeddings

def embedding_score(query: str, chunk_embeddings: list, chunks: list):
  """
  Compute embedding similarity score between a query and a list of text chunks.
  """
  # if not chunk_embeddings:
  #   return []

  query_embedding = EMBEDDING_MODEL.encode(inputs = [query], show_progress_bar = False, convert_to_numpy = True)
  scores = EMBEDDING_MODEL.similarity(query_embedding, chunk_embeddings)

  # unsorted_with_doc_ids = [{"id": doc_id, "source": chunks[doc_id]["source"], "chunk": chunks[doc_id]["chunk"], "embedding_score": float(scores.flatten()[doc_id])} for doc_id in range(len(chunks))]
  unsorted_with_doc_ids = [{ **item, "embedding_score": float(scores.flatten()[doc_id]) } for doc_id, item in enumerate(chunks)]
  return unsorted_with_doc_ids

def reciprocal_rank_fusion(bm25_results, embedding_results, k=60, id_key="id", bm25_key="bm25_score", embedding_key="embedding_score"):
  """
  Compute Reciprocal Rank Fusion (RRF) score for a set of documents based on their ranks in two different scoring systems (BM25 and embedding similarity).

  Args:
    bm25_results (list of dict): A list of dictionaries containing BM25 scores for each document, with keys including 'id' and the specified bm25_key.
    embedding_results (list of dict): A list of dictionaries containing embedding similarity scores for each document, with keys including 'id' and the specified embedding_key.
    k (int, optional): A constant used in the RRF formula to control the influence of ranks. Defaults to 60.
    id_key (str, optional): The key used to identify documents in the input lists. Defaults to "id".
    bm25_key (str, optional): The key used to access BM25 scores in the bm25_results. Defaults to "bm25_score".
    embedding_key (str, optional): The key used to access embedding similarity scores in the embedding_results. Defaults to "embedding_score".
  Returns:
    A list of dictionaries representing the fused results, sorted by their RRF score in descending order. Each dictionary includes the document id, original BM25 and embedding scores, and the computed RRF score.
  """
  bm25_ranked = sorted(bm25_results, key=lambda x: x.get(bm25_key, 0.0), reverse=True)
  emb_ranked = sorted(embedding_results, key=lambda x: x.get(embedding_key, 0.0), reverse=True)

  bm25_rank_by_id = {item[id_key]: rank for rank, item in enumerate(bm25_ranked, start=1)}
  emb_rank_by_id = {item[id_key]: rank for rank, item in enumerate(emb_ranked, start=1)}

  # bm25_by_id = {item[id_key]: item for item in bm25_results}
  # emb_by_id = {item[id_key]: item for item in embedding_results}
  bm25_by_id = {item[id_key]: item for item in bm25_results}
  emb_by_id = {item[id_key]: item for item in embedding_results}

  all_ids = set(bm25_rank_by_id) | set(emb_rank_by_id)
  fused = []

  for doc_id in all_ids:
    r_bm25 = bm25_rank_by_id.get(doc_id)
    r_emb = emb_rank_by_id.get(doc_id)

    rrf_score = 0.0
    if r_bm25 is not None:
      rrf_score += 1.0 / (k + r_bm25)
    if r_emb is not None:
      rrf_score += 1.0 / (k + r_emb)

    # base = bm25_by_id.get(doc_id, emb_by_id.get(doc_id, {}))
    fused.append(
      {
        "id": doc_id,
        **bm25_results[doc_id],
        # bm25_key: bm25_by_id.get(doc_id, {}).get(bm25_key),
        bm25_key: bm25_by_id.get(doc_id, {}).get(bm25_key, 0.0),
        # embedding_key: emb_by_id.get(doc_id, {}).get(embedding_key),
        embedding_key: emb_by_id.get(doc_id, {}).get(embedding_key, 0.0),
        "rrf_score": rrf_score,
      }
    )

  return sorted(fused, key=lambda x: x["rrf_score"], reverse=True)

def reciprocal_rank_fusion_new(*list_of_list_ranks_system, K = 60):
  """
  Fuse rank from multiple IR systems using Reciprocal Rank Fusion.
  
  Args:
  * list_of_list_ranks_system: Ranked results from different IR system.
  K (int): A constant used in the RRF formula (default is 60).
  
  Returns:
  Tuple of list of sorted documents by score and sorted documents
  """
  # Dictionary to store RRF mapping
  rrf_map = defaultdict(float)

  # Calculate RRF score for each result in each list
  for rank_list in list_of_list_ranks_system:
    for rank, item in enumerate(rank_list, 1):
      rrf_map[item] += 1 / (rank + K)

  # Sort items based on their RRF scores in descending order
  sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse = True)

  # Return tuple of list of sorted documents by score and sorted documents
  return sorted_items, [item for item, score in sorted_items]