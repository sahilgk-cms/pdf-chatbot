import streamlit as st
import sys
import os
from constants import *
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.chat import *
from utils.vector_embeddings import *
import asyncio

try:
    loop = asyncio.get_running_loop()
    loop.close()
except RuntimeError:
    pass  # No loop was running, so nothing to close


st.title("OAK V1 - OpenAI Embeddings & LLM")



if "chat_engine_oak" not in st.session_state:
    st.session_state["chat_engine_oak"] = None

with st.spinner("Initialising chat engine...."):
    try:
        vector_index_oak = load_vector_index_from_weaviate(weaviate_index = WEAVIATE_INDEX_FOR_OAK)
        chat_engine_oak = create_chat_engine(vector_index_oak)
        st.session_state["chat_engine_oak"] = chat_engine_oak
        st.success("Chat engine initialized successfully")
    except Exception as e:
        st.error(f"Error: {e}")


if st.session_state["chat_engine_oak"]:
    if "chat_history_oak_v1" not in st.session_state:
        st.session_state["chat_history_oak_v1"] = []

    query = st.chat_input("Ask a question")

    if query:
        st.session_state["chat_history_oak_v1"].append({"role": "user", "message": query})
        #try:
        answer, source_dict = qa_chat_excel(chat_engine = st.session_state["chat_engine_oak"],
                        query = query)
        
        if answer and source_dict:
            source_text = "\n\n**Sources:**\n"
            for doc, sheets in source_dict.items():
                source_text += f"{doc}: (Sheets: {', '.join(sheets)})\n"

            full_response = f"{answer}\n\n```{source_text}```"
            st.session_state["chat_history_oak_v1"].append({"role": "assistant", "message": full_response})
        else:
            st.session_state["chat_history_oak_v1"].append({"role": "assistant",
                                                        "message": "Answer not found in any of the documents"})

        #except Exception as e:
            #st.error(f"Error: {e}")

    for chat in st.session_state["chat_history_oak_v1"]:
        if chat["role"] == "user":
            st.chat_message("user").write(f"**Query:** {chat["message"]}")
        elif chat["role"] == "assistant":
            st.chat_message("assistant").write(f"**Answer:** {chat["message"]}")
      
    