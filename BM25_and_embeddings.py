from collections import defaultdict
from constants import ORACLE_SKILLS, SKIP_DIRS, MARKDOWN_LANGUAGE
from pathlib import Path
import bm25s
import Stemmer
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

headers = [ ("#", "h1"), ("##", "h2"), ("###", "h3") ]

section_splitter = MarkdownHeaderTextSplitter(headers_to_split_on = headers, strip_headers = True)

child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name = "cl100k_base",
    chunk_size = 500,
    chunk_overlap = 40,
    separators = ["\n\n", "\n", ". ", " ", ""],
)

EMBEDDING_MODEL = SentenceTransformer(model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2")

def langchain_chunk_by_sections(text: str, source_path: str):
  sections = section_splitter.split_text(text)
  chunks = []

  for section_id, doc in enumerate(sections):
    h1 = doc.metadata.get("h1")
    h2 = doc.metadata.get("h2")
    h3 = doc.metadata.get("h3")

    path_parts = [x for x in [h1, h2, h3] if x]
    # header_path = " > ".join(path_parts)

    body = doc.page_content.strip()

    for chunk_id, piece in enumerate(child_splitter.split_text(body)):
      chunks.append({ "source": source_path, "section_id": section_id, "chunk_id": chunk_id, "header_path": path_parts, "chunk": piece })

  return chunks

def chunks_from_markdown_files():
  """
  Read markdown files from the ORACLE_SKILLS directory, split them into sections and chunks, and return a list of chunks with metadata.
  Each chunk is represented as a dictionary containing the source file path, section id, chunk id, header path (list of headers), and the chunk text.
  Returns:
    A list of dictionaries, each representing a chunk of text with associated metadata.
  """
  docs_root = Path(ORACLE_SKILLS)
  markdown_files = []
  for path in sorted(docs_root.rglob("*.md")):
    if any(part in SKIP_DIRS for part in path.parts):
      continue
    if path.is_file():
      markdown_files.append(path)
  chunks = []
  i = 0
  for path in markdown_files:
    chunk_sections = langchain_chunk_by_sections(text = path.read_text(encoding = "utf-8", errors = "ignore").strip(), source_path = str(path))
    for chunk in chunk_sections:
      chunks.append({"id": i, **chunk })
      i += 1
  return chunks

def bm25_retriever_from_chunks(chunks: list):
  """
  Create a BM25 retriever from a list of text chunks.

  Args:
    chunks (list of dict): A list of dictionaries, each containing a 'chunk' key with the text to be indexed.

  Returns:
    A BM25 retriever object that can be used to retrieve relevant chunks based on a query.
  """
  # Stemmer
  stemmer = Stemmer.Stemmer(MARKDOWN_LANGUAGE)
  corpus_tokens = bm25s.tokenize(texts = ["\n".join(x for x in ["\n".join(item.get("header_path", [])), item.get("chunk", ""), Path(item.get("source","")).name] if x) for item in chunks], show_progress = False, stopwords = "en", stemmer = stemmer)
  retriever = bm25s.BM25(corpus = chunks)
  retriever.index(corpus = corpus_tokens)
  return retriever

def bm25_score(query: str, chunks: list, retriever, top_k: int = 50):
  """
  Compute BM25 score between a query and a list of text chunks.

  Args:
    query (str): The input query to compare against the chunks.
    chunks (list of dict): A list of dictionaries, each containing a 'chunk' key with the text to be scored.
    retriever: BM25 retriever object.
    stemmer: Stemmer object used for tokenization.
    top_k (int, optional): The number of top results to return. Defaults to 20.

  Returns: A tuple of (results, scores) where results is a list of chunks sorted  by relevance to the query, and scores is a list of corresponding BM25 scores.
  """
  if not chunks:
    return []

  # Stemmer
  # stemmer = Stemmer.Stemmer("english")
  # corpus_tokens = bm25s.tokenize(texts = ["\n".join(x for x in ["\n".join(item.get("header_path", [])), item.get("chunk", ""), Path(item.get("source","")).name] if x) for item in chunks], show_progress = False, stopwords = "en", stemmer = stemmer)
  # retriever = bm25s.BM25(corpus = chunks)
  # retriever.index(corpus = corpus_tokens)

  # You can now search the corpus with a query
  query_tokens = bm25s.tokenize(texts = [query], show_progress = False, stemmer = Stemmer.Stemmer(MARKDOWN_LANGUAGE))

  results, scores = retriever.retrieve(query_tokens = query_tokens, show_progress = False, k = top_k)

  bm25_score = []
  for i in range(results.shape[1]):
    result, score = results[0, i], scores[0, i]
    # bm25_score.append({ "id": int(result['id']), "file": str(result['source']), "chunk": str(result['chunk']), "bm25_score": float(score) })
    bm25_score.append({ **result, "bm25_score": float(score) })

  # The bm25s function is returning on the top_k results but I need the full chunks list unsorted with their respective score (0 if not scored)
  ranking_chunks = [{ **chunk, "bm25_score": 0.0} for chunk in chunks]

  for item in bm25_score:
    ranking_chunks[item.get("id")]["bm25_score"] = item.get("bm25_score", 0.0)

  return ranking_chunks

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