#   Virtual DBA

The objective of this project is to create a Virtual DBA agent that will use [Oracle Skills repository](https://github.com/oracle/skills) and inject expertise in the prompt before answering to Oracle database user's question.

## Used libraries

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

## How to install and play

Either take the zip release or git clone.

You must put the Oracle Skills repository inside the folder where you have installed the application.

Install the required Python dependencies with:
```bash
pip install -r requirements.txt
```


Finally run the Virtual DBA applcaition with:
```bash
streamlit run virtual_dba.py
```