import os
from glob import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from PyPDF2 import PdfReader
import textwrap
import pathlib

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in .env file")

embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Locate 'docs' folder from project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
docs_path = os.path.join(project_root, "docs")
pdf_files = glob(os.path.join(docs_path, "*.pdf"))

all_chunks = []
all_metadata = []

# Extract basic metadata from PDF
def extract_pdf_metadata(pdf_path):
    reader = PdfReader(pdf_path)
    info = reader.metadata or {}

    # Always use filename as fallback title
    title = os.path.basename(pdf_path).replace(".pdf", "").replace("_", " ")

    return {
        "filename": os.path.basename(pdf_path),
        "author": info.get("/Author", "Unknown Author"),
        "title": info.get("/Title", title.strip()),
        "journal": info.get("/Subject", "Unknown Journal")
    }

# Load and annotate PDF pages
for file_path in pdf_files:
    metadata = extract_pdf_metadata(file_path)
    all_metadata.append(metadata)

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    for page in pages:
        appended_text = textwrap.dedent(f"""\
            [Author: {metadata['author']}, Title: {metadata['title']}, Journal: {metadata['journal']}]
            {page.page_content.strip()}
        """)
        all_chunks.append(Document(
            page_content=appended_text,
            metadata=metadata
        ))

print(f"\nLoaded and annotated {len(all_chunks)} chunks from {len(pdf_files)} PDFs.")

# Preview sample chunks
print("\nSample Annotated Chunks:")
for i, chunk in enumerate(all_chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print("Metadata:", chunk.metadata)
    print("Content preview:\n", chunk.page_content[:500], "...\n")

# Embed in safe batches
BATCH_SIZE = 100
vectorstore = None

for i in range(0, len(all_chunks), BATCH_SIZE):
    batch = all_chunks[i:i + BATCH_SIZE]
    print(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)...")
    if vectorstore is None:
        vectorstore = FAISS.from_documents(batch, embedding_model)
    else:
        vectorstore.add_documents(batch)

# Save vectorstore after all batches
vectorstore_path = os.path.join(project_root, "src", "vectorstore")
os.makedirs(vectorstore_path, exist_ok=True)
vectorstore.save_local(os.path.join(vectorstore_path, "faiss_index"))
print(f"Vectorstore saved to: {os.path.join(vectorstore_path, 'faiss_index')}")
print("Absolute path:", pathlib.Path(vectorstore_path).resolve())

# Save metadata summary
metadata_path = os.path.join(vectorstore_path, "metadata_summary.txt")
with open(metadata_path, "w", encoding="utf-8") as f:
    for meta in all_metadata:
        f.write(f"Filename: {meta['filename']}\n")
        f.write(f"  Author : {meta['author']}\n")
        f.write(f"  Title  : {meta['title']}\n")
        f.write(f"  Journal: {meta['journal']}\n")
        f.write("-" * 40 + "\n")

print(f"Metadata saved to: {metadata_path}")
