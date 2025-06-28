import os
from glob import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in .env file")

embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Dynamically resolve absolute path to root-level 'docs' folder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
docs_path = os.path.join(project_root, "docs")

# Load all PDFs
pdf_files = glob(os.path.join(docs_path, "*.pdf"))
documents = []

for file_path in pdf_files:
    loader = PyPDFLoader(file_path)
    documents.extend(loader.load())

print(f"Loaded {len(documents)} pages from {len(pdf_files)} PDFs.")

# Build vectorstore
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local("vectorstore/faiss_index")

print("Vectorstore built and saved successfully.")
