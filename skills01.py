from pathlib import Path
from ollama import ChatResponse, chat

OLLAMA_MODEL = "llama3.2:latest"
skills_file = Path("D:/Yannick/Development/python/ollama/oracle-db-skills-main/skills/performance/awr-reports.md")

skills_text = skills_file.read_text(encoding="utf-8")

# question = "Extract the main technical skills, tools, and programming languages from this file."
# question = "what is DB Time"

while True:
  question = input("\nAsk a question about the markdown files (or 'exit'): ").strip()
  if question.lower() == "exit":
    break
  response: ChatResponse = chat(
    model = OLLAMA_MODEL,
    messages=[
    {
    "role": "system",
    "content": "You analyze Markdown files and extract structured information."
    },
    {
    "role": "user",
    "content": f"""
    Here is a Markdown file containing skills information:

    --- BEGIN FILE ---
    {skills_text}
    --- END FILE ---

    Question:
    {question}

    Return the answer in Markdown.
    """
    }
    ]
  )

  print("Answer:\n" + response["message"]["content"])