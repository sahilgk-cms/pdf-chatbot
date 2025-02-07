from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader, SimpleKeywordTableIndex, VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
import openai
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from typing import List
import pandas as pd
import os
import sys
import llama_index
import nest_asyncio
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
nest_asyncio.apply()


EMBEDDING_MODEL = "text-embedding-3-small"
MODEL_NAME = "gpt-3.5-turbo"
EMBED_DIMENSION = 512
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 50

#### Retrieving credentials from .env file
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#### Retrieving credentials from Streamlit Community
OPENAI_API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    raise ValueError("OPENAI_API_KEY is not set. Please provide a valid API key.")


Settings.embed_model = OpenAIEmbedding(model = EMBEDDING_MODEL, dimensions = EMBED_DIMENSION)

def convert_pdf_files_into_llamaindex_docs(folder_path: str,
                                           chunk_size: int,
                                           chunk_overlap: int) -> List[llama_index.core.schema.Document]:
    '''
  This function converts the pdfs present in the folder path into llamaindex documents.
  Args:
    pdf_folder_path, chunk_size, chunk_overlap
  Returns:
    list of llamaindex documents
   '''
    pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]

    langchain_documents = []
    for pdf_file in pdf_files:
        pdf_file_path = os.path.join(folder_path, pdf_file)
        pdf_loader = PyPDFLoader(pdf_file_path)
        pdf_documents = pdf_loader.load()
        langchain_documents.extend(pdf_documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    split_documents = text_splitter.split(langchain_documents)

    llamaindex_documents = []
    for doc in split_documents:
        llamaindex_doc = Document.from_langchain_format(doc)
        llamaindex_documents.append(llamaindex_doc)
    
    return llamaindex_documents


def embedd_documents_into_vector_index_and_save_to_faiss(documents: List[llama_index.core.schema.Document],
                                                            file_path: str):
  '''
  This function embedds the llamaindex documents into FAISS database & stores then at the given filepath.
  Args:
    llamaindex documents, filepath
  Returns:
    saves the embeddings into faiss database locally
   '''
  d = EMBED_DIMENSION
  faiss_index = faiss.IndexFlatL2(d)
  vector_store = FaissVectorStore(faiss_index = faiss_index)
  storage_context = StorageContext.from_defaults(vector_store=vector_store)
  vector_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
  vector_index.storage_context.persist(file_path)


def load_vector_index_from_faiss(file_path: str) -> VectorStoreIndex:
  '''
  This function loads the embeddings from the FAISS database & creates a vector index
  Args:
    filepath of the saved FAISS database
  Returns:
    vector index
   '''
  vector_store = FaissVectorStore.from_persist_dir(file_path)
  storage_context = StorageContext.from_defaults(vector_store = vector_store, persist_dir = file_path)
  vector_index = load_index_from_storage(storage_context = storage_context)
  return vector_index


def create_chat_engine(vector_index: VectorStoreIndex):
   '''
  This function create dthe chat engine
  Args:
    vector store index
  Returns:
    chat engine (current based on OpenAI)
  '''
   llm = OpenAI(model = MODEL_NAME)
   chat_engine = vector_index.as_chat_engine(chat_mode = "best", llm = llm, verbose = True)
   return chat_engine


def qa_chat(chat_engine, query: str) -> str:
   '''
  This functions returns the answer & source of the answer to a question
  Args:
     chat engine, query
  Returns:
    answer & its source to the given query
  '''
   result = chat_engine.chat(query)
   answer = result.response
   source_nodes = result.source_nodes
   source_dict = {}

   for i in range(0, len(source_nodes)):
      source_document = os.path.basename(source_nodes[i].metadata["source"])
      page_number = source_nodes[i].metadata["page_label"]

      if source_document not in source_dict:
        source_dict[source_document] = []

      if page_number not in source_dict[source_document]:
        source_dict[source_document].append(page_number)
      
   return answer, source_dict



