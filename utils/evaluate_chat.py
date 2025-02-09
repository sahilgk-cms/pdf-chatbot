from typing import List
import openai
import faiss
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from typing import List
import numpy as np
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
OPENAI_API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    raise ValueError("OPENAI_API_KEY is not set. Please provide a valid API key.")


def check_faithfulness(query: str, response: llama_index.core.chat_engine.types.AgentChatResponse) -> bool:
    '''
  This function evaluated the faithfulness of the response.
  Args:
    query, response
  Returns:
    True if the response is faithful else False
    '''
    llm = OpenAI(model = MODEL_NAME)
    faithfulness_eval = FaithfulnessEvaluator(llm = llm)
    return faithfulness_eval.evaluate_response(query, response).passing


def check_relevancy(query: str, response: llama_index.core.chat_engine.types.AgentChatResponse) -> bool:
  '''
  This function evaluated the relevancy of the response.
  Args:
    query, response
  Returns:
    True if the response is relevant else False
  '''
  llm = OpenAI(model_name = "gpt-3.5-turbo-0613")
  relevancy_eval = RelevancyEvaluator(llm = llm)
  return relevancy_eval.evaluate_response(query, response).passing
