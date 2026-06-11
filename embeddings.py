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