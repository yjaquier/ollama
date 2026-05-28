from constants import ORACLE_SKILLS, SKIP_DIRS, MARKDOWN_LANGUAGE
from pathlib import Path
import bm25s
import Stemmer
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

headers = [ ("#", "h1"), ("##", "h2"), ("###", "h3") ]

section_splitter = MarkdownHeaderTextSplitter(headers_to_split_on = headers, strip_headers = True)

child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name = "cl100k_base",
    chunk_size = 500,
    chunk_overlap = 40,
    separators = ["\n\n", "\n", ". ", " ", ""],
)

def langchain_chunk_by_sections(text: str, source_path: str):
  """
  Split a markdown text into sections based on headers and then into smaller chunks. Each chunk is associated with metadata including the source file path, section id, chunk id, and header path.
  
  Args:
    text (str): The input markdown text to be split into sections and chunks.
    source_path (str): The path to the source markdown file.

  Returns:
    A list of dictionaries, each representing a chunk of text with associated metadata.
  """
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

  Returns:
    A tuple of (results, scores) where results is a list of chunks sorted  by relevance to the query, and scores is a list of corresponding BM25 scores.
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