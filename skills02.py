from pathlib import Path
from typing import Any
from ollama import ChatResponse, chat, AsyncClient
import asyncio

OLLAMA_MODEL = "llama3.2:latest"
# OLLAMA_MODEL = "codegemma:2b"
skills_file = Path("D:/Yannick/Development/python/ollama/oracle-db-skills-main/skills/performance/awr-reports.md")
SKILLS_DIRECTORY = Path("D:/Yannick/Development/python/ollama/oracle-db-skills-main/")
MAX_INGEST_CHARS = 80000
CHUNK_SIZE = 12000

skills_text = skills_file.read_text(encoding="utf-8")

def load_recursive_skills_markdown_files(root: Path, pattern: str = "*.md", max_chars: int = MAX_INGEST_CHARS) -> str:
  """
  Recursively load Markdown files from the specified directory, concatenating their contents into a single string.
  The function respects a maximum character limit to avoid overwhelming the model with too much input.
  """
  parts = []
  total = 0

  for path in sorted(root.rglob(pattern)):
    print(path)
    try:
      text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
      print(f"Skipping {path}: {exc}")
      continue

    block = f"\n\n===== FILE: {path} =====\n{text}"
    if total + len(block) > max_chars:
      remaining = max_chars - total
      if remaining > 0:
        parts.append(block[:remaining])
      break

    parts.append(block)
    total += len(block)

  return "".join(parts).strip()

def load_skills_markdown_file(file: Path, max_chars: int = MAX_INGEST_CHARS) -> str:
  """
  Recursively load Markdown files from the specified directory, concatenating their contents into a single string.
  The function respects a maximum character limit to avoid overwhelming the model with too much input.
  """
  parts = []
  total = 0

  print(file)
  try:
    text = file.read_text(encoding="utf-8", errors="ignore")
  except Exception as exc:
    print(f"Skipping {file}: {exc}")
    return ""

  block = f"\n\n===== FILE: {file} =====\n{text}"
  if total + len(block) > max_chars:
    remaining = max_chars - total
    if remaining > 0:
      parts.append(block[:remaining])

  parts.append(block)
  total += len(block)

  return "".join(parts).strip()

def split_text(text: str, chunk_size: int):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

async def ollama_chat_by_chunk(ollama: AsyncClient, full_skills: str, question: str) -> ChatResponse:
  partial_answers = []

  for idx, chunk in enumerate(split_text(full_skills, CHUNK_SIZE), start=1):
    response = await ollama.chat(
        model = OLLAMA_MODEL,
        messages = [
          {
            "role": "system",
            "content": "Extract information only from the provided text chunk."
          },
          {
            "role": "user",
            "content": f"""
              This is chunk {idx} of repository Markdown content.

              Text:
              {chunk}

              Question:
              {question}

              Return a concise bullet list.
              """
          }
        ]
    )
    partial_answers.append(response["message"]["content"])

    final_response = await ollama.chat(
    model = OLLAMA_MODEL,
    messages = [
        {
          "role": "system",
          "content": "Merge partial extraction results into one final deduplicated answer."
        },
        {
          "role": "user",
          "content": f"""
            Here are partial answers extracted from multiple Markdown chunks:

            {chr(10).join(partial_answers)}

            Create one final deduplicated Markdown answer grouped by category.
            """
        }
      ]
    )
  return final_response

async def ollama_chat(ollama: AsyncClient, full_skills: str, question: str) -> ChatResponse:
  response = await ollama.chat(
  model = OLLAMA_MODEL,
  messages = [
      {
        "role": "system",
        "content": "Extract information only from the provided text skill."
      },
      {
        "role": "user",
        "content": f"""
          Here is a Markdown file containing skills information:

          --- BEGIN FILE ---
          {full_skills}
          --- END FILE ---

          Question:
          {question}

          Return the answer in Markdown.
          """
      }
    ]
  )
  return response

async def main():
  # question = "Extract the main technical skills, tools, and programming languages from this file."
  # question = "what is DB Time"

  summary_skill = ""
  # full_skills = load_skills_markdown_files(SKILLS_DIRECTORY)
  # summary_skill += load_skills_markdown_file(SKILLS_DIRECTORY / "AGENTS.md")
  # summary_skill += load_skills_markdown_file(SKILLS_DIRECTORY / "SKILLS.md")
  summary_skill += load_skills_markdown_file(SKILLS_DIRECTORY / "SKILL.md")
  # summary_skill += load_skills_markdown_file(SKILLS_DIRECTORY / "skills-index.md")

  ollama = AsyncClient()

  while True:
    question = input("\nAsk a question about the markdown files (or 'exit'): ").strip()
    if question.lower() == "exit":
      break
    response: ChatResponse = await ollama_chat(ollama, summary_skill, question)

    print(f"Answer generated in seconds: {response.total_duration / 1e9:.2f}(s)") # Total duration in nanoseconds / 1e9 = seconds
    print("Answer:\n" + response.message.content)

if __name__ == "__main__":
    asyncio.run(main())