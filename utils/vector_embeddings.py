from typing import List
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
import openai
import weaviate
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import Settings
from typing import List
import pandas as pd
import os
import sys
import llama_index
import nest_asyncio
from constants import *
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
nest_asyncio.apply()

#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

OPENAI_API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]
WEAVIATE_API_KEY = st.secrets["weaviate"]["WEAVIATE_API_KEY"]

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    raise ValueError("OPENAI_API_KEY is not set. Please provide a valid API key.")

if WEAVIATE_API_KEY:
    os.environ["WEAVIATE_API_KEY"] = WEAVIATE_API_KEY
else:
    raise ValueError("WEAVIATE_API_KEY is not set. Please provide a valid API key.")


Settings.embed_model = OpenAIEmbedding(model = EMBEDDING_MODEL, dimensions = EMBED_DIMENSION)

client = weaviate.connect_to_wcs(cluster_url = WEAVIATE_URL, 
                                 auth_credentials = weaviate.auth.AuthApiKey(WEAVIATE_API_KEY))


def embedd_documents_into_vector_index_and_save_to_weaviate(documents:List[llama_index.core.schema.Document],
                                                             weaviate_index: str):
  vector_store = WeaviateVectorStore(index_name = weaviate_index)
  storage_context = StorageContext.from_defaults(vector_store = vector_store)
  index = VectorStoreIndex.from_documents(documents, storage_context = storage_context)


def load_vector_index_from_weaviate(weaviate_index: str) -> VectorStoreIndex:
  vector_store = WeaviateVectorStore(weaviate_client = client,
                                     index_name = weaviate_index)
  vector_index = VectorStoreIndex.from_vector_store(vector_store = vector_store)
  return vector_index
