import logging
from pathlib import Path
import pandas as pd
import io
import queue
import threading
import streamlit as st
from constants import LOG_FILE, OLLAMA_MODEL, OLLAMA_MODEL_LIST, DEFAULT_ACTIVE_CONNECTION_NAME, ORACLE_SKILLS
from services import KnowledgeService, SqlService

# =============================================================================
# Streamlit page config
# =============================================================================
# https://emojipedia.org/ for the icon of your choice
page_title = "Virtual DBA"
page_icon = "🪄"
st.set_page_config(
    page_title=page_title,
    page_icon=page_icon,
    layout="wide"
)

# =============================================================================
# Logging
# =============================================================================
def setup_logging() -> logging.Logger:
  logger = logging.getLogger("agent")
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

  return logger

LOGGER = setup_logging()

# =============================================================================
# Cached services
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_knowledge_service() -> KnowledgeService:
  return KnowledgeService(LOGGER)

@st.cache_resource(show_spinner=False)
def get_sql_service() -> SqlService:
  return SqlService(LOGGER)

sql_service = get_sql_service()
knowledge_service = get_knowledge_service()

# =============================================================================
# Session state
# =============================================================================
def init_session_state() -> None:
  defaults = {
    "messages": [],
    # "cache_data": None,
    # "cache_key": None,
    "chunks": None, # List of chunkcs of all markdown files
    "retriever": None, # The BM25 retriever
    "ollama_model": OLLAMA_MODEL,
    "cancel_requested": False,
    "active_connection_name": DEFAULT_ACTIVE_CONNECTION_NAME,
    "is_busy": False,
    "queued_question": None,
    "chat_input_value": "",
    "rag_thread": None,
    "rag_queue": None,
    "rag_stop_event": None,
    "rag_running": False,
    "rag_stream_text": "",
    "rag_finished": False,
  }
  for key, value in defaults.items():
    if key not in st.session_state:
      st.session_state[key] = value

init_session_state()

# =============================================================================
# RAG thread helpers
# Running the RAG process in a thread in mandatory to be able to cancel the call.
# Or due to the native way of working of Streamlit when you click the "Cancel Request" buton the information does not go along the current running instance?
# So the need of threading to share information between threads.
# =============================================================================
def run_rag_in_thread(question: str, ollama_model: str, chunks, retriever, q: queue.Queue, stop_event: threading.Event):
  def on_chunk(full_text: str, last_chunk: str):
    q.put({
      "type": "chunk",
      "full_text": full_text,
    })

  def is_cancel_requested() -> bool:
    return stop_event.is_set()

  try:
    response = knowledge_service.handle_question(
      ollama_model = ollama_model,
      question = question,
      chunks = chunks,
      retriever = retriever,
      on_chunk = on_chunk,
      is_cancel_requested=is_cancel_requested,
    )
    q.put({
      "type": "done",
      "response": response,
    })
  except Exception as exc:
    LOGGER.exception("RAG worker failed: %s", exc)
    q.put({
      "type": "error",
      "error": str(exc),
    })

def start_rag_request(question: str) -> None:
  q = queue.Queue()
  stop_event = threading.Event()

  st.session_state.rag_queue = q
  st.session_state.rag_stop_event = stop_event
  st.session_state.rag_stream_text = ""
  st.session_state.rag_running = True
  st.session_state.rag_finished = False
  st.session_state.is_busy = True
  st.session_state.cancel_requested = False

  thread = threading.Thread(
    target=run_rag_in_thread,
    args=(
      question,
      st.session_state.ollama_model,
      st.session_state.chunks,
      st.session_state.retriever,
      q,
      stop_event,
    ),
    daemon=True,
  )
  thread.start()
  st.session_state.rag_thread = thread

def clear_rag_state() -> None:
  st.session_state.rag_thread = None
  st.session_state.rag_queue = None
  st.session_state.rag_stop_event = None
  st.session_state.rag_running = False
  st.session_state.rag_stream_text = ""
  st.session_state.is_busy = False
  st.session_state.cancel_requested = False

def drain_rag_queue() -> None:
  q = st.session_state.rag_queue
  if q is None:
    return

  while True:
    try:
      item = q.get_nowait()
    except queue.Empty:
      break

    item_type = item.get("type")

    if item_type == "chunk":
      st.session_state.rag_stream_text = item.get("full_text", "")

    elif item_type == "done":
      response = item["response"]
      st.session_state.messages.append({
        "role": "assistant",
        "message_type": "rag",
        "content": response,
      })
      clear_rag_state()
      st.session_state.rag_finished = True

    elif item_type == "error":
      st.session_state.messages.append({
        "role": "assistant",
        "message_type": "markdown",
        "content": f"Error while processing request: {item['error']}",
      })
      clear_rag_state()
      st.session_state.rag_finished = True

@st.fragment(run_every="1s")
def rag_live_refresh():
  drain_rag_queue()

  if st.session_state.rag_finished:
    st.session_state.rag_finished = False
    st.rerun()

  if st.session_state.rag_running:
    with st.chat_message("assistant"):
      if st.session_state.rag_stream_text:
        st.markdown(st.session_state.rag_stream_text + "▌")
      else:
        st.markdown(f"Generating answer with {st.session_state.ollama_model}...")

      if st.session_state.cancel_requested:
        st.caption("Cancel requested...")

# =============================================================================
# UI
# =============================================================================
st.title(page_icon + " " + page_title)
st.caption("Ask questions on your markdown SQL knowledge base, execute SQL, or inspect snippets.")

with st.sidebar:
  st.header("Configuration")

  current_model_index = OLLAMA_MODEL_LIST.index(st.session_state.ollama_model) if st.session_state.ollama_model in OLLAMA_MODEL_LIST else 0
  
  st.session_state.ollama_model = st.selectbox(label = "Which model to use?", options = OLLAMA_MODEL_LIST, index = current_model_index, disabled = st.session_state.is_busy)

  docs_root_str = st.text_input("Skills root folder", value = ORACLE_SKILLS, disabled=st.session_state.is_busy)

  connection_name = st.text_input("SQLcl connection name", value = st.session_state.active_connection_name, disabled=st.session_state.is_busy)

  st.divider()
  st.header("Cache control")
  if st.button("Rebuild cache", disabled = st.session_state.is_busy):
    try:
      docs_root = Path(docs_root_str).resolve()
      with st.spinner("Generating chunks and BM25..."):
        st.session_state.chunks = knowledge_service.initialize_chunks(docs_root)
        st.session_state.retriever = knowledge_service.bm25_retriever_from_chunks(st.session_state.chunks)
      st.success("Chunks and BM25 generated successfully.")
    except Exception as exc:
      LOGGER.exception(f"Failed to load cache: {exc}")
      st.error(f"Failed to load cache: {exc}")

  st.divider()
  st.header("Chat control")
  col1, col2 = st.columns(2)
  with col1:
    if st.button("Clear history", disabled = st.session_state.is_busy):
      st.session_state.messages = []
      st.rerun()
  with col2:
    if st.button("Cancel Request", disabled = not(st.session_state.rag_running)):
      LOGGER.info("Clicked on 'Cancel Request'")
      st.session_state.cancel_requested = True
      stop_event = st.session_state.get("rag_stop_event")
      if stop_event is not None:
        stop_event.set()

  st.divider()
  st.header("Prompt examples")
  st.code('How to generate an AWR report in HTML format ?', language="text")
  st.code('execute("SELECT * FROM DBA_HIST_SNAPSHOT")', language="text")
  st.code('execute("SELECT * FROM v$session")', language="text")
  # st.code('show_snippet("performance\\awr-reports.md::snippet_6")', language="text")

# =============================================================================
# Initialization of chunks and BM25 retriever
# =============================================================================
if st.session_state.chunks is None or st.session_state.retriever is None:
  try:
    docs_root = Path(docs_root_str).resolve()
    LOGGER.info(f"Initialization started from docs root: {docs_root}")
    with st.spinner("Generating chunks and BM25..."):
      st.session_state.chunks = knowledge_service.initialize_chunks(docs_root)
      st.session_state.retriever = knowledge_service.bm25_retriever_from_chunks(st.session_state.chunks)
      LOGGER.info(f"Initialization complete from docs root: {docs_root}")
    # st.session_state.cache_key = current_cache_key
  except Exception as exc:
    LOGGER.exception(f"Failed to auto-load cache: {exc}")
    st.error(f"Failed to load cache: {exc}")
    st.stop()


# =============================================================================
# Chat history
# This function is super important as Streamlit re-process the whole history
# each time you submit a new question.
# So you must handle correctly all the different stuffs you have put inside
# For example I may have an st.expander a duration or whatever.
# message template in {"role": "assistant", "message_type": "markdown | rag | csv | sql | ...", "content": "the content"}
# Then content template is:
#   markdown: str
#   rag: {"answer": full_response, "ollama_model": ollama_model, "duration_ns": total_duration_ns, "was_cancelled": True | False, "relevant_files": relevant_files}
#   csv: str
#   sql: str
# 
# =============================================================================
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    content = message.get("content", "")

    if isinstance(content, str): # markdown, csv or sql
      if message.get("message_type") == "csv":
        df = pd.read_csv(io.StringIO(content.strip()))
        st.dataframe(df, width = 'stretch')      
      elif message.get("message_type") == "markdown":
        st.markdown(content)
      elif message.get("message_type") == "sql":
        st.code(content, language="sql")
      else:
        st.markdown(content)
    elif isinstance(content, dict): #rag
      answer = content.get("answer", "")
      was_cancelled = content.get("was_cancelled", False)
      if answer is not None and not was_cancelled:
        st.markdown(answer)
        st.markdown("**Retrieval Log**")
        with st.expander("Relevant files", expanded=False):
          for item in content.get("relevant_files", []):
            st.write(f"File: {item['source']}, Section: {" > ".join(item.get("header_path", [])) if isinstance(item.get("header_path", []), list) else str(item.get("header_path", []))}")
      duration_ns = content.get("duration_ns")
      ollama_model = content.get("ollama_model")
      if duration_ns is not None and ollama_model is not None:
        st.caption(f"Generated with {ollama_model} in {float(duration_ns) / 1e9:.2f} seconds")
    else:
      st.markdown(str(content)) # Desperate case or role = "user"

# =============================================================================
# Live RAG streaming area
# =============================================================================
rag_live_refresh()

# =============================================================================
# Chat input
# =============================================================================
def queue_question() -> None:
  if st.session_state.is_busy:
    return

  question = st.session_state.get("chat_input_value", "").strip()
  if not question:
    return

  st.session_state.queued_question = question
  st.session_state.is_busy = True
  st.session_state.cancel_requested = False
  st.session_state.chat_input_value = ""
  
st.chat_input(
  # placeholder='Ask your question, or use execute("...") / show_snippet("...")',
  placeholder='Ask your question, or use execute("...")',
  key="chat_input_value",
  on_submit=queue_question,
  disabled=st.session_state.is_busy
)

# if question and not st.session_state.is_busy:
#   st.session_state.is_busy = True
if st.session_state.is_busy and st.session_state.queued_question:
  question = st.session_state.queued_question
  st.session_state.queued_question = None # To avoid re-processing by mistake the same question...

  st.session_state.messages.append({"role": "user", "content": question})

  try:
    if question.lower().startswith("execute("): # Execute a SQL case
      with st.spinner("Executing SQL..."):
        response = sql_service.handle_execute_command(question = question, connection_name = connection_name.strip())
      if response is not None:
        df = pd.read_csv(io.StringIO(str(response).strip()))
        st.dataframe(df, width = 'stretch')
        st.session_state.messages.append({"role": "assistant", "message_type": "csv", "content": response})
      st.session_state.is_busy = False
      # st.rerun()
    # elif question.lower().startswith("show_snippet("): # Show a snippet case
    #   with st.spinner("Loading snippet..."):
    #     response = knowledge_service.handle_show_snippet_command(question = question, cache_data = st.session_state.cache_data)
    #   if response:
    #     st.markdown(response)
    #     st.session_state.messages.append({"role": "assistant", "message_type": "markdown", "content": response})
    else: # Normal "RAG" question so accessing the Ollama model
      start_rag_request(question)
      # st.rerun()

  except Exception as exc:
    error_message = f"Error while processing request: {exc}"
    LOGGER.exception(error_message)
    st.session_state.messages.append({"role": "assistant", "message_type": "markdown", "content": error_message})
    st.session_state.is_busy = False
    # st.rerun()
  finally:
    st.rerun()

