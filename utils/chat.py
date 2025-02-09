from typing import List
from llama_index.core import  VectorStoreIndex
import openai
from llama_index.llms.openai import OpenAI
from typing import List
import numpy as np
import os
import sys
import llama_index
import nest_asyncio
from constants import *
from utils.evaluate_chat import *
from dotenv import load_dotenv


load_dotenv()
nest_asyncio.apply()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    raise ValueError("OPENAI_API_KEY is not set. Please provide a valid API key.")



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
    faithfulness = check_faithfulness(query, result)
    relevancy = check_relevancy(query, result)
    source_dict = {}
    scores = []
   
    for i in range(0, len(source_nodes)):
      source_document = os.path.basename(source_nodes[i].metadata["source"])
      page_number = source_nodes[i].metadata["page_label"]
      scores.append(source_nodes[i].score)

      if source_document not in source_dict:
        source_dict[source_document] = []

      if page_number not in source_dict[source_document]:
        source_dict[source_document].append(page_number)

    scores = np.array(scores)
    if scores.mean() >= NODE_THRESHOLD and faithfulness and relevancy:
        return answer, source_dict
    else:
        answer = None
        source_dict = None
        return answer, source_dict