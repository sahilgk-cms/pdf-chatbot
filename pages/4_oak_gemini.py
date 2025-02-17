import streamlit as st
from utils.files_processing import *
from utils.chat import *
import warnings
warnings.simplefilter("ignore", ResourceWarning)
import asyncio

# Close any existing event loop
try:
    loop = asyncio.get_running_loop()
    loop.close()
except RuntimeError:
    pass  # No loop was running, so nothing to close


st.title("OAK V2 - Gemini & basic prompt template framework")


if "oak_data" not in st.session_state:
    st.session_state["oak_data"] = None

if "chat_history_oak_v2" not in st.session_state:
    st.session_state["chat_history_oak_v2"] = []

if st.session_state["oak_data"] is None:
    with st.spinner("Loading pdf data...."):
        text = load_dict_from_json(file_path = "oak_excel_dict.json")
        st.session_state["oak_data"] = text



query = st.chat_input("Ask a question")

if query:
    st.session_state["chat_history_oak_v2"].append({"role": "user", "message": query})
    try:
        result = qa_chat2(text = st.session_state["oak_data"], query = query)
        answer = result["answer"]
        source = result["source"]
        full_response = f"\n**Answer:** {answer}\n\n**Source:** {source}"
        st.session_state["chat_history_oak_v2"].append({"role": "assistant", "message": full_response})
    except Exception as e:
        st.error(f"Error: {e}")


for chat in st.session_state["chat_history_oak_v2"]:
    if chat["role"] == "user":
        st.chat_message("user").write(f"**Query:** {chat["message"]}")
    elif chat["role"] == "assistant":
        st.chat_message("assistant").write(chat["message"])




