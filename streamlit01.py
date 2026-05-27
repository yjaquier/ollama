import streamlit as st
from ollama import chat, ChatResponse

OLLAMA_MODEL = "llama3.2:latest"

# Initialize chat history
if "messages" not in st.session_state:
  st.session_state.messages = [{"role": "assistant", "content": "Hello, how can I help ?"}]
  st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# Clear history button
if st.button("Clear history"):
  st.session_state.messages = [{"role": "assistant", "content": "Hello, how can I help ?"}]
  st.session_state.messages = []
  st.rerun()

# Accept user input
if prompt := st.chat_input(placeholder = "Send a message..."):
  # Add user message to chat history
  st.session_state.messages.append({"role": "user", "content": prompt})
  # Display user message in chat message container
  with st.chat_message("user"):
    st.markdown(prompt)

  # Display assistant response in chat message container
  with st.chat_message("assistant"):
    message_placeholder = st.empty()
    duration_placeholder = st.empty()
    full_response = ""
    total_duration_ns = 0
    try:
      response: ChatResponse = chat(
        model = OLLAMA_MODEL,
        messages = st.session_state.messages,
        stream = True,
      )
      for chunk in response:
        full_response += chunk.message.content
        message_placeholder.markdown(full_response + "▌")
        if chunk.total_duration:
          total_duration_ns = chunk.total_duration
    except Exception as exc:
      full_response = f"Ollama request failed: {exc}"

    message_placeholder.markdown(full_response)
    duration_placeholder.caption(f"Total duration in seconds : {total_duration_ns / 1e9:.2f}s")
    
  # Add assistant response to chat history
  st.session_state.messages.append({"role": "assistant", "content": full_response})
