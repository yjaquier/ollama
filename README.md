#   Virtual DBA

The objective of this project is to create an Virtual DBA agent that will use [Oracle Skills repository](https://github.com/oracle/skills) and inject expertise in the prompt before answering to Oracle database user's question.

The technology stack is made of:
- Python 3.14
- [Ollama](https://ollama.com/)
- [Streamlit](https://streamlit.io/)

To inject the good chunks of markdown documents in my Retrieval-Augmented Generation (RAG) I have used:
- [Ollama Python Library](https://github.com/ollama/ollama-python) to connect to my offline local Ollama instance from Python
- [LangChain](https://www.langchain.com/) to split markdown files in chunks based on section (MarkdownHeaderTextSplitter from langchain-text-splitters project)
- [BM25S project](https://github.com/xhluca/bm25s) for BM25 algorithm
- [SentenceTransformer](https://sbert.net/index.html) for embedding algorithm
- The Reciprocal Rank Fusion (RRF) procedre is home made
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) for connection to SQLcl MCP server