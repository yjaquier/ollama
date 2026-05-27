import hashlib
import json
import logging
import re
from pathlib import Path
from ollama import Client
from constants import CACHE_DIR, FILE_SUMMARIES_FILE, SNIPPET_STORE_FILE, GLOBAL_SUMMARY_FILE, RETRIEVAL_INDEX_FILE, SKIP_DIRS, KNOWN_DATABASES, KNOWN_TOOLS, KNOWN_SQL_TOPICS, THINK_TO_PARSE

def parse_json_loose(raw: str, fallback: dict) -> dict:
  """
  Parse JSON in a tolerant way.

  The local model may sometimes return:
  - plain JSON
  - JSON wrapped in markdown fences
  - extra text around a JSON object

  This function tries a few safe parsing strategies and returns
  the provided fallback if parsing still fails.
  """
  raw = raw.strip()

  try:
    return json.loads(raw)
  except json.JSONDecodeError:
    pass

  if raw.startswith("```"):
    lines = raw.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
      raw = "\n".join(lines[1:-1]).strip()
      try:
        return json.loads(raw)
      except json.JSONDecodeError:
        pass

  start = raw.find("{")
  end = raw.rfind("}")
  if start != -1 and end != -1 and end > start:
    candidate = raw[start:end + 1]
    try:
      return json.loads(candidate)
    except json.JSONDecodeError:
      pass

  return fallback


def build_or_update_cache(ollama_model: str, client:Client, docs_root: Path, rebuild_cache: bool, max_file_chars: int, logger: logging.Logger):
  """
  Scan markdown files, extract SQL fenced blocks, summarize markdown prose with Ollama,
  and persist cache files.

  Cache content:
  - file summaries
  - SQL snippet store
  - global summary
  - retrieval index

  Files are reused from cache when their content hash did not change.
  """
  CACHE_DIR.mkdir(parents=True, exist_ok=True)

  old_file_summaries = []
  old_snippet_store = []

  if FILE_SUMMARIES_FILE.exists():
    old_file_summaries = json.loads(FILE_SUMMARIES_FILE.read_text(encoding="utf-8"))
  if SNIPPET_STORE_FILE.exists():
    old_snippet_store = json.loads(SNIPPET_STORE_FILE.read_text(encoding="utf-8"))

  old_summary_by_file = {item["file"]: item for item in old_file_summaries}
  old_snippets_by_file = {}
  for snippet in old_snippet_store:
    old_snippets_by_file.setdefault(snippet["file"], []).append(snippet)

  markdown_files = []
  for path in sorted(docs_root.rglob("*.md")):
    if any(part in SKIP_DIRS for part in path.parts):
      continue
    if path.is_file():
      markdown_files.append(path)

  logger.info("Markdown files found: %s", len(markdown_files))

  file_summaries = []
  snippet_store = []

  # Support executable code snippets in fenced blocks for many languages, not just SQL.
  # Example:
  # ```sql
  # ...
  # ```
  # ```java
  # ...
  # ```
  # etc.
  # The language is captured as group 1, the code as group 2.
  # sql_block_pattern = re.compile(r"```sql\s*\r?\n(.*?)\r?\n```", re.IGNORECASE | re.DOTALL)
  sql_block_pattern = re.compile(r"^[ \t]*```([a-zA-Z0-9_+-]+)\s*\r?\n(.*?)\r?\n[ \t]*```", re.IGNORECASE | re.DOTALL | re.MULTILINE)

  for path in markdown_files:
    rel_path = str(path.relative_to(docs_root))
    directory = str(path.parent.relative_to(docs_root))
    text = path.read_text(encoding="utf-8", errors="ignore")[:max_file_chars]
    file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    cached_summary = old_summary_by_file.get(rel_path)
    if cached_summary and cached_summary.get("_hash") == file_hash and not rebuild_cache:
      logger.info(f"[CACHE] Reusing {rel_path} file cache")
      file_summaries.append(cached_summary)
      for snippet in old_snippets_by_file.get(rel_path, []):
        snippet_store.append(snippet)
      continue

    logger.info(f"[INGEST] Processing {rel_path} file with {ollama_model} model")

    # Extract SQL snippets from fenced markdown blocks.
    sql_snippets = []
    for idx, match in enumerate(sql_block_pattern.finditer(text), start=1):
      langage = match.group(1).strip()
      snippet_text = match.group(2).strip()
      # snippet_text = match.group(1).strip()
      start_line = text[:match.start()].count("\n") + 1
      end_line = text[:match.end()].count("\n") + 1

      lowered = snippet_text.strip().lower()

      # Keep kind classification simple and readable.
      if langage == 'sql':
        if lowered.startswith("select") or lowered.startswith("with"):
          kind = "select"
        elif lowered.startswith("show ") or lowered.startswith("info ") or lowered.startswith("desc ") or lowered.startswith("describe ") or lowered.startswith("set "):
          kind = "sqlcl"
        elif lowered.startswith("begin") or lowered.startswith("declare"):
          kind = "plsql"
        elif lowered.startswith("insert") or lowered.startswith("update") or lowered.startswith("delete") or lowered.startswith("merge"):
          kind = "dml"
        elif lowered.startswith("create") or lowered.startswith("alter") or lowered.startswith("drop") or lowered.startswith("truncate") or lowered.startswith("grant") or lowered.startswith("revoke"):
          kind = "ddl"
        else:
          kind = "sql_like"
      else:
        kind = "other"

      keywords = []
      for token in re.findall(r"[A-Za-z0-9_\.]+", snippet_text.lower()):
        if token not in {
          "select", "from", "where", "and", "or", "group", "order", "by", "join",
          "left", "right", "inner", "outer", "on", "as", "with", "begin", "end",
          "insert", "update", "delete", "create", "alter", "drop", "set", "show"
        } and len(token) >= 3:
          keywords.append(token)
      keywords = sorted(set(keywords))[:25]

      first_line = snippet_text.splitlines()[0].strip() if snippet_text.splitlines() else f"snippet_{idx}"
      if len(first_line) > 100:
        first_line = first_line[:97] + "..."

      sql_snippets.append({
        "id": f"{rel_path}::snippet_{idx}",
        "file": rel_path,
        "directory": directory,
        "kind": kind,
        "langage": langage,
        "title": first_line,
        "content": snippet_text,
        "line_start": start_line,
        "line_end": end_line,
        "keywords": keywords
      })

    snippet_store.extend(sql_snippets)

    # Remove SQL fenced blocks before semantic summarization.
    # This is important because raw code often causes malformed JSON output.
    text_without_sql = sql_block_pattern.sub("", text).strip()

    prompt = f"""
Return ONLY valid JSON.

Task:
Extract explicit technical information from the markdown prose below.

Use only the provided content.
Do not use outside knowledge.
Do not leave fields empty if matching terms are clearly present in the text.

Expected JSON schema:
{{
"title": "short title if visible, otherwise empty string",
"summary": "one short sentence describing the file",
"keywords": ["short keywords explicitly present in the prose"],
"skills": ["short technical capabilities explicitly mentioned"]
}}

Extraction rules:
- Fill arrays with strings only
- Prefer explicit terms found in the text
- Do not invent information
- "skills" means technical capabilities or areas of expertise explicitly described in the prose
- Keep the summary short and factual
- Return JSON only
- Title and summary can be extracted from OVERVIEW section of markdown prose

Markdown prose:
{text_without_sql}
""".strip()

    try:
      response = client.chat(
          model = ollama_model,
          think = THINK_TO_PARSE,
          messages = [ {"role": "system", "content": "Return valid JSON only."}, {"role": "user", "content": prompt} ],
          format = "json"
        )
      
      raw = response["message"]["content"].strip()
    except Exception:
      logger.exception("Ollama summarization failed for %s", rel_path)
      raw = ""

    logger.info("Raw summary output for %s:\n%s", rel_path, raw)
    databases = set()
    tools = set()
    sql_topics = set()

    for pattern, label in KNOWN_DATABASES.items():
      if re.search(rf"\b{re.escape(pattern)}\b", lowered):
        databases.add(label)

    for pattern, label in KNOWN_TOOLS.items():
      if re.search(rf"\b{re.escape(pattern)}\b", lowered):
        tools.add(label)

    for pattern, label in KNOWN_SQL_TOPICS.items():
      if re.search(rf"\b{re.escape(pattern)}\b", lowered):
        sql_topics.add(label)

    summary = parse_json_loose(raw, {
      "title": "",
      "summary": "",
      "keywords": [],
      "skills": []
    })

    summary = {
      "file": rel_path,
      "directory": directory,
      "title": summary.get("title", ""),
      "summary": summary.get("summary", ""),
      "keywords": summary.get("keywords", []) if isinstance(summary.get("keywords", []), list) else [],
      "skills": summary.get("skills", []) if isinstance(summary.get("skills", []), list) else [],
      "databases": sorted(databases),
      "tools": sorted(tools),
      "sql_topics": sorted(sql_topics),
      "has_sql_snippets": bool(sql_snippets),
      "_hash": file_hash
    }

    file_summaries.append(summary)

  # Build simple retrieval indexes based on keywords.
  keyword_to_files = {}
  keyword_to_snippets = {}

  for summary in file_summaries:
    file_path = summary["file"]
    terms = []
    terms.extend(summary.get("keywords", []))
    terms.extend(summary.get("skills", []))
    terms.extend(summary.get("databases", []))
    terms.extend(summary.get("tools", []))
    terms.extend(summary.get("sql_topics", []))
    terms.extend(re.findall(r"[A-Za-z0-9_\.]+", summary.get("summary", "").lower()))

    for term in terms:
      key = str(term).strip().lower()
      if len(key) < 2:
        continue
      keyword_to_files.setdefault(key, set()).add(file_path)

  for snippet in snippet_store:
    snippet_id = snippet["id"]
    terms = []
    terms.extend(snippet.get("keywords", []))
    terms.extend(re.findall(r"[A-Za-z0-9_\.]+", snippet.get("title", "").lower()))
    terms.append(snippet.get("kind", ""))

    for token in re.findall(r"[A-Za-z0-9_\.]+", snippet["content"].lower()):
      if len(token) >= 3:
        terms.append(token)

    for term in terms:
      key = str(term).strip().lower()
      if len(key) < 2:
        continue
      keyword_to_snippets.setdefault(key, set()).add(snippet_id)

  retrieval_index = {
    "keyword_to_files": {k: sorted(v) for k, v in keyword_to_files.items()},
    "keyword_to_snippets": {k: sorted(v) for k, v in keyword_to_snippets.items()}
  }

  global_summary = {
      "skills": sorted(set(
          str(x).strip()
          for summary in file_summaries
          for x in summary.get("skills", [])
          if str(x).strip()
      )),
      "databases": sorted(set(
          str(x).strip()
          for summary in file_summaries
          for x in summary.get("databases", [])
          if str(x).strip()
      )),
      "tools": sorted(set(
          str(x).strip()
          for summary in file_summaries
          for x in summary.get("tools", [])
          if str(x).strip()
      )),
      "sql_topics": sorted(set(
          str(x).strip()
          for summary in file_summaries
          for x in summary.get("sql_topics", [])
          if str(x).strip()
      )),
      "files": [
          {
              "file": s["file"],
              "directory": s["directory"],
              "title": s["title"],
              "summary": s["summary"],
              "has_sql_snippets": s["has_sql_snippets"]
          }
          for s in file_summaries
      ]
  }

  FILE_SUMMARIES_FILE.write_text(json.dumps(file_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
  SNIPPET_STORE_FILE.write_text(json.dumps(snippet_store, indent=2, ensure_ascii=False), encoding="utf-8")
  GLOBAL_SUMMARY_FILE.write_text(json.dumps(global_summary, indent=2, ensure_ascii=False), encoding="utf-8")
  RETRIEVAL_INDEX_FILE.write_text(json.dumps(retrieval_index, indent=2, ensure_ascii=False), encoding="utf-8")

  logger.info("Cache updated successfully")
  return file_summaries, snippet_store, retrieval_index, global_summary
