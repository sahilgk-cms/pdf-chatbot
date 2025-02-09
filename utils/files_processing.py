from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import  Document
from typing import List
import os
import llama_index
from dotenv import load_dotenv


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