#   Virtual DBA

The objective of this project is to create an agent that will use [Oracle Skills repository](https://github.com/oracle/skills) and inject expertise in the prompt before answering to the user.

The technology stack is made of:
- Python 3.14
- [Ollama](https://ollama.com/)
- [Streamlit](https://streamlit.io/)

To inject the good chunks of markdown documents in my Retrieval-Augmented Generation (RAG) I have used:
- [BM25S project](https://github.com/xhluca/bm25s) for BM25 algorithm
- [SentenceTransformer](https://sbert.net/index.html) for embedding algorithm