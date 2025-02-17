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


################################################ EXCEL TO JSON ################################################################################

def get_single_sheet_inside_excel(file_path: str, sheet_name: str):
  '''
  This function returns the dataframe from execl file & a given sheet
  Args:
    file path, sheet name
  Returns:
    dataframe of that given sheet
    '''
  df = pd.read_excel(file_path, sheet_name)
  columns_to_be_dropped = [col for col in df.columns if col.startswith("Unnamed")]
  df = df.drop(columns_to_be_dropped, axis = 1)
  return df


def convert_dataframe_into_dict_and_llamaindex_docs(file_path: str,
                                                    sheet_name: str) -> Tuple[List[dict], List[llama_index.core.schema.Document]]:
    '''
    This function converts the single sheet inside excel file into dictionary records (for chat prompt) & llamaindex documents (for embeddings)
    Args:
      file path, sheet name
    Returns:
      dictionary records (for chat prompt) & llamaindex documents (for embeddings)
    '''
    df = get_single_sheet_inside_excel(file_path, sheet_name)

    #this is for Gemini prompt
    dict_ = df.to_dict(orient = "records")

    #this is for LlamaIndex documents & OpenAI
    documents = []
    for entry in dict_:
      cleaned_entry = {k.strip(): v.strip() if isinstance(v, str) else v for k, v in entry.items()}
      text = "\n".join(f"{key}:{value}" for key, value in cleaned_entry.items())
      doc = Document(text=text)
      doc.metadata = {"file_name": os.path.basename(file_path),
                     "sheet": sheet_name}
      documents.append(doc)

    return dict_, documents


def custom_serializer(obj):
    """
    Serialize non-JSON serializable objects, such as datetime.
    Args:
        obj (any): The object to serialize.
    Returns:
        str: An ISO 8601 formatted string if the object is a datetime instance.
    Raises:
        TypeError: If the object type is not supported for serialization.
    """
    if isinstance(obj, datetime):
      return obj.isoformat()
    raise TypeError(f"Type {type(obj)} is not serializable")


def save_dict_to_json(input_dict: dict, file_path: str):
  '''
  This function saves the dictionary to JSON file
  Args:
    input dictionary, file path
  Returns:
    None
  '''
  with open(file_path, "w") as f:
    json.dump(input_dict, f, default=custom_serializer)

def load_dict_from_json(file_path: str) -> dict:
  '''
  This function loads the dictionary from json file
  Args:
    file path
  Returns:
    dictionary
  '''
  with open(file_path, "r") as f:
    loaded_dict  = json.load(f)
  return loaded_dict
