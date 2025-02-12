import streamlit as st
import sys
import os
from constants import *
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.chat import *
from utils.vector_embeddings import *


st.title("Chatbot")
st.header("Used OpenAI Embeddings & LLM")


if "chat_engine" not in st.session_state:
    st.session_state["chat_engine"] = None

with st.spinner("Initialising chat engine...."):
    try:
        vector_index = load_vector_index_from_weaviate(weaviate_index = WEAVIATE_INDEX)
        chat_engine = create_chat_engine(vector_index)
        st.session_state["chat_engine"] = chat_engine
        st.success("Chat engine initialized successfully")
    except Exception as e:
        st.error(f"Error: {e}")


if st.session_state["chat_engine"]:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    query = st.chat_input("Ask a question")

    if query:
        st.session_state["chat_history"].append({"role": "user", "message": query})
        try:
            answer, source_dict = qa_chat(chat_engine = st.session_state["chat_engine"],
                            query = query)
            
            if answer and source_dict:
                source_text = "\n\n**Sources:**\n"
                for doc, pages in source_dict.items():
                    source_text += f"{doc}: (Pages: {', '.join(pages)})\n"

                full_response = f"{answer}\n\n```{source_text}```"
                st.session_state["chat_history"].append({"role": "assistant", "message": full_response})
            else:
                st.session_state["chat_history"].append({"role": "assistant",
                                                            "message": "Answer not found in any of the documents"})

        except Exception as e:
             st.error(f"Error: {e}")

    for chat in st.session_state["chat_history"]:
        if chat["role"] == "user":
            st.chat_message("user").write(f"**Query:** {chat["message"]}")
        elif chat["role"] == "assistant":
            st.chat_message("assistant").write(f"**Answer:** {chat["message"]}")
      
    
