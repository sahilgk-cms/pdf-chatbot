from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import  Document
from typing import List
import os
import llama_index
from pypdf import PdfReader
from dotenv import load_dotenv

############################################## LLAMAINDEX DOCUMENTS ################################################################

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

################################################ PDF TO TEXT ################################################################################

def convert_pdf_into_text(folder_path: str) -> str:
  '''
  This function converts all pdf data into text string
  Args:
    folder path containing all pdfs
  Returns:
    text string
  '''
  pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]
  print(f"PDF Files in Google Drive Folder: {pdf_files}")

  text = ""
  for pdf_file in pdf_files:
    pdf_path = os.path.join(folder_path, pdf_file)
    reader = PdfReader(pdf_path)

    for page in reader.pages:
      text += page.extract_text()

  return text

def save_string_to_textfile(text: str, filepath: str):
  '''
  This function saves string to text file
  Args:
    string
  Returns:
    None
  '''
  with open(filepath, "w", encoding = "utf-8") as f:
    f.write(text)


def load_string_from_textfile(filepath: str) -> str:
  '''
  This function loads string from a text file
  Args:
    Text file path
  Returns:
    String
  '''
  with open(filepath, "r", encoding="utf-8") as f:
    text = f.read()
  return text
