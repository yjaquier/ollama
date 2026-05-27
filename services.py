import asyncio
import logging
from pathlib import Path
from typing import Optional
from ollama import Client
from constants import CACHE_DIR
from parse_markdown_files import build_or_update_cache
from answer_and_execute import answer_question, execute_sqlcl_mcp, show_snippet

from BM25_and_embeddings import chunks_from_markdown_files, bm25_retriever_from_chunks


class KnowledgeService:
  def __init__(self, logger: logging.Logger):
      self.logger = logger
      self.client = Client()

  # def load_cache(self, ollama_model: str, docs_root: Path, rebuild_cache: bool, max_file_chars: int):
  #   CACHE_DIR.mkdir(parents=True, exist_ok=True)

  #   if not docs_root.exists():
  #     raise FileNotFoundError(f"Docs root does not exist: {docs_root}")

  #   self.logger.info("Loading cache from docs root: %s", docs_root)

  #   return build_or_update_cache(
  #     ollama_model = ollama_model,
  #     client = self.client,
  #     docs_root=docs_root,
  #     rebuild_cache=rebuild_cache,
  #     max_file_chars=max_file_chars,
  #     logger = self.logger
  #   )
  def initialize_chunks(self, docs_root: Path):
    if not docs_root.exists():
      raise FileNotFoundError(f"Docs root does not exist: {docs_root}")

    self.logger.info("Initializing chunks from docs root: %s", docs_root)

    return chunks_from_markdown_files()
  
  def bm25_retriever_from_chunks(self, chunks):
    self.logger.info("Initializing BM25 retriever from chunks")
    return bm25_retriever_from_chunks(chunks)
  
  def handle_question(self, ollama_model: str, question: str, chunks, retriever, on_chunk = None, is_cancel_requested = None):

    self.logger.info("Question: %s", question)
    self.logger.info(f"Selected Ollama model: {ollama_model}")

    return answer_question(
      ollama_model = ollama_model,
      client = self.client,
      question = question,
      chunks = chunks,
      retriever = retriever,
      logger = self.logger,
      on_chunk = on_chunk,
      is_cancel_requested = is_cancel_requested
    )

  def handle_show_snippet_command(self, question: str, cache_data) -> Optional[str]:
    snippet_id = self.extract_quoted_argument(question, "show_snippet")
    if snippet_id is None:
      raise ValueError('Invalid show_snippet(...) format. Use show_snippet("snippet_id").')

    snippet_id = snippet_id.strip()
    if not snippet_id:
      raise ValueError("Empty snippet id in show_snippet(...).")

    _, snippet_store, _, _ = cache_data
    snippet_text, snippet_text_flat = show_snippet(snippet_store, snippet_id)

    return (
      f"### Snippet id: `{snippet_id}`\n\n"
      f"**Formatted snippet:**\n\n"
      f"```sql\n{snippet_text}\n```\n\n"
      f"**Flat snippet for copy/paste:**\n\n"
      f"```sql\n{snippet_text_flat}\n```"
    )

    # message_placeholder.markdown(response)
    # return response

  @staticmethod
  def extract_quoted_argument(command: str, function_name: str) -> Optional[str]:
    prefix = f"{function_name}("
    text = command.strip()

    if not text.lower().startswith(prefix.lower()):
      return None

    if not text.endswith(")"):
      return None

    inner = text[len(prefix):-1].strip()
    if len(inner) < 2:
      return None

    if (inner[0] == '"' and inner[-1] == '"') or (inner[0] == "'" and inner[-1] == "'"):
      return inner[1:-1]

    return None


class SqlService:
  def __init__(self, logger: logging.Logger):
      self.logger = logger

  def handle_execute_command(self, question: str, connection_name: str):
    sql_statement = KnowledgeService.extract_quoted_argument(question, "execute")
    if sql_statement is None:
      raise ValueError('Invalid execute(...) format. Use execute("SELECT ...").')

    sql_statement = sql_statement.strip()
    if not sql_statement:
      raise ValueError("Empty SQL statement in execute(...).")

    if not connection_name:
      raise ValueError("A connection name is required to execute SQL.")

    self.logger.info("Executing SQL: %s", sql_statement)

    # ensure_connected(connection_name)

    # We must use asyncio.run to run the execute_sqlcl_mcp asnc function because call to SQLcl MCP Server must be executed with async library functions...
    result = asyncio.run(execute_sqlcl_mcp(tool_command="run-sql", arguments={"connection_name": connection_name, "sql": sql_statement}, logger=self.logger))
    return result
  