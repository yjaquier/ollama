from contextlib import AsyncExitStack
import json
import logging
import re
from typing import Any
from ollama import Client
from mcp import ClientSession, stdio_client
from constants import THINK_TO_ANSWER, SERVER_PARAMS
from BM25 import bm25_score

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# The Ollama call
# ---------------------------------------------------------------------------------------------------------------------------------------------------
def answer_question(
  ollama_model: str,
  client: Client,
  question: str,
  chunks: list,
  retriever: Any,
  logger: logging.Logger,
  on_chunk = None,
  is_cancel_requested = None
):
  """
  Retrieve relevant files/snippets for a question, answer using summaries,
  and optionally execute selected SQL through SQLcl MCP.

  Safe DB execution policy:
  - direct execution allowed only for snippet kinds: select, sqlcl
  - generated SQL must start with SELECT or WITH
  - unsafe SQL keywords are blocked
  """
  bm25_score_dict = bm25_score(question, chunks, retriever, 100)

  # We need to build a prompt like this with the top-n ranked chunks
  # source: skills-main/db/performance/awr-reports.md
  # section: awr reports — automatic workload repository > overview
  # content:
  # the automatic workload repository (awr) is oracle's built-in ...

  NUMBER_OF_CHUNK_TO_INCLUDE_IN_PROMPT = 5
  logger.info(f"=== TOP {NUMBER_OF_CHUNK_TO_INCLUDE_IN_PROMPT} RELEVANT FILES ===")
  retrieved_context = ""
  sorted_bm25_score_dict = sorted(bm25_score_dict, key=lambda x: x.get("bm25_score", 0.0), reverse=True)
  for item in sorted_bm25_score_dict[:NUMBER_OF_CHUNK_TO_INCLUDE_IN_PROMPT]:
    logger.info(f"- {item.get('source')}")
    retrieved_context += f"[{item.get('id')}]\nsource: {item.get('source')}\nsection: {" > ".join(item.get('header_path', []))}\ncontent: {item.get('chunk')}\n\n"

  answer_prompt = f"""
Question:
{question}

Retrieved context:
{retrieved_context}

Instructions:
- Answer in Markdown
- Answer using only the retrieved context above.
- If the context does not fully answer the question, say what is missing.
""".strip()

  full_response = ""
  total_duration_ns = 0
  was_cancelled = False
  try:
    response = client.chat(
      model = ollama_model,
      think = THINK_TO_ANSWER,
      messages = [
        {"role": "system", "content": "Answer only from the provided summaries and snippet metadata."},
        {"role": "user", "content": answer_prompt}
      ],
      stream = True
    )
    for chunk in response:
      if is_cancel_requested is not None and is_cancel_requested():
        logger.info("Response generation cancelled by user.")
        full_response += "\n\n*Response generation was cancelled by the user.*"
        was_cancelled = True
        break
      # logger.info(f"Value of is_cancel_requested: {is_cancel_requested()}")
      text = getattr(chunk.message, "content", "") or ""
      if text:
        full_response += text
        if on_chunk is not None:
          on_chunk(full_response, text)
      if getattr(chunk, "total_duration", None):
        total_duration_ns = chunk.total_duration
  except Exception as exc:
    full_response = f"Ollama request failed: {exc}"

  logger.info(f"Response generated with {ollama_model} in {total_duration_ns / 1e9:.2f} seconds")
  # return full_response, total_duration_ns

  return {
    "answer": full_response,
    "ollama_model": ollama_model,
    "duration_ns": total_duration_ns,
    "was_cancelled": was_cancelled,
    "relevant_files": sorted_bm25_score_dict[:NUMBER_OF_CHUNK_TO_INCLUDE_IN_PROMPT]
  }

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Call to SQLcl MCP server
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# def _normalize_mcp_result(result: Any) -> str:
#   if result is None:
#     return ""

#   # if isinstance(result, str):
#   #   return result

#   # if isinstance(result, (dict, list)):
#   #   try:
#   #     return json.dumps(result, indent=2, ensure_ascii=False)
#   #   except Exception:
#   #     return str(result)

#   if hasattr(result, "content"):
#     try:
#       content = result.content
#       if isinstance(content, str):
#         return content
#       return json.dumps(content, indent=2, ensure_ascii=False)
#     except Exception:
#       return str(result)

#   return str(result)

def _normalize_mcp_result(result: Any) -> str:
    """
    Normalize MCP tool results into readable text.

    Handles common MCP result shapes such as:
    - plain strings
    - dict/list
    - objects with:
        - content: [TextContent(...), ...]
        - structuredContent
        - isError
    """

    if result is None:
      return ""

    # Plain string
    if isinstance(result, str):
      return result

    # Plain JSON-like structures
    if isinstance(result, (dict, list)):
      try:
        return json.dumps(result, indent=2, ensure_ascii=False)
      except Exception:
        return str(result)

    # MCP-style object fields
    content = getattr(result, "content", None)
    structured_content = getattr(result, "structuredContent", None)
    is_error = getattr(result, "isError", False)

    extracted_parts = []

    # Extract content list items
    if content:
      for item in content:
        # Most useful MCP case: TextContent(... text='...')
        text_value = getattr(item, "text", None)
        if text_value is not None:
          extracted_parts.append(str(text_value))
          continue

        # Fallback if item itself is a string
        if isinstance(item, str):
          extracted_parts.append(item)
          continue

        # Fallback for json-like item
        if isinstance(item, (dict, list)):
          try:
            extracted_parts.append(json.dumps(item, indent=2, ensure_ascii=False))
          except Exception:
            extracted_parts.append(str(item))
          continue

        # Generic fallback
        extracted_parts.append(str(item))

    # If no textual content found, try structured content
    if not extracted_parts and structured_content is not None:
      if isinstance(structured_content, str):
        extracted_parts.append(structured_content)
      else:
        try:
          extracted_parts.append(json.dumps(structured_content, indent=2, ensure_ascii=False))
        except Exception:
          extracted_parts.append(str(structured_content))

    # Final fallback
    if not extracted_parts:
      extracted_parts.append(str(result))

    output = "\n".join(part for part in extracted_parts if part is not None).strip()

    if is_error:
      return f"Error from MCP tool:\n{output}"

    return output

# Even if Streamlit behaves like a reactive UI script the library to call to SQLcl MCP server is only available as an async set of functions
# So this function must be async and you must call it as such in the main program with asyncio.run()
# This is the easiest implementation I have found
# I also create the on-time session in this procedure to simplify the Streamlit main program
async def execute_sqlcl_mcp(
    tool_command: str,
    arguments: dict | None,
    logger: logging.Logger,
) -> str:
  """
  Retrieve relevant files/snippets for a question, answer using summaries,
  and optionally execute selected SQL through SQLcl MCP.

  Safe DB execution policy:
  - direct execution allowed only for snippet kinds: select, sqlcl
  - generated SQL must start with SELECT or WITH
  - unsafe SQL keywords are blocked
  """
  arguments = arguments or {}
  logger.info(f"Calling tool command: {tool_command} with arguments: {json.dumps(arguments)}")

  async with AsyncExitStack() as exit_stack:
    read, write = await exit_stack.enter_async_context(stdio_client(SERVER_PARAMS))
    session = await exit_stack.enter_async_context(ClientSession(read, write))

    await session.initialize()

    # logger.info("MCP session initialized for action=%s", tool_command)

    # For list-connections or disconnect tool_command there is no arguments
    if tool_command in ["list-connections", "disconnect"]:
      try:
        result = await session.call_tool(tool_command, {})
        return _normalize_mcp_result(result)
      except Exception as exception:
        logger.info(exception)

    # For connect the argument is { "connection_name": "testpdb01" }
    if tool_command == "connect":
      try:
        result = await session.call_tool(tool_command, arguments)
        return _normalize_mcp_result(result)
      except Exception as exception:
        logger.info(exception)

    # For the run-sql case the connection to the databse must be setup for the same session
    # the tool_command is "run-sql"
    # so the payload should be like { "connection_name": "testpdb01", "sql": "sql_statement" }
    if tool_command == "run-sql":
      try:
        result = await session.call_tool("connect", { "connection_name": arguments.get("connection_name", "") })
        result = await session.call_tool(tool_command, { "sql": arguments.get("sql", "") } )
      except Exception as exception:
        logger.info(exception)

    # logger.info(result)
    return _normalize_mcp_result(result)

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Return the code of a snippet, one original version and one version flatered for easier copy/paste
# ---------------------------------------------------------------------------------------------------------------------------------------------------
def show_snippet(snippet_store, snippet_id):
  """
  Show the code of a snippet
  return the exact snippet code and a flat version to allow easy copy/paste for execute("...") function
  """
  if not snippet_id:
    return None

  for snippet in snippet_store:
    if isinstance(snippet, dict) and snippet.get("id") == snippet_id:
      snippet_flat = re.sub(
        r"\s+",
        " ",
        " ".join(
          line for line in snippet.get("content", "").splitlines()
          if not line.lstrip().startswith("--")
        ),
      ).strip().replace("\n", " ")  # Remove SQL comments and collapse whitespace
      return snippet.get("content", ""), snippet_flat

  return None