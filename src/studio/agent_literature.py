import os
import json
from dotenv import load_dotenv

from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, MessagesState, StateGraph

# Load environment variables
load_dotenv()

# Helper function to load a prompt from the 'src/prompts' directory
def load_prompt(file_name: str) -> str:
    """Loads a prompt template from a file."""
    # This logic is copied from your working file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    prompt_path = os.path.join(project_root, "src", "prompts", file_name)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_path}")
        raise

# Path and Vectorstore Setup - Copied from your working file
project_src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
vectorstore_path = os.path.join(project_src_root, "vectorstore/faiss_index")

print("Looking for index.faiss at:", os.path.join(vectorstore_path, "index.faiss"))
assert os.path.exists(os.path.join(vectorstore_path, "index.faiss")), "index.faiss not found"
assert os.path.exists(os.path.join(vectorstore_path, "index.pkl")), "index.pkl not found"

# Load vectorstore
vectorstore = FAISS.load_local(
    folder_path=vectorstore_path,
    index_name="index",
    embeddings=OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever()

# Load LLM and prompt
llm_extract = ChatOpenAI(model="gpt-4.1", temperature=0.2)
extract_prompt_text = load_prompt("extract_known_sites_lit_prompt.txt")
extract_prompt = PromptTemplate.from_template(extract_prompt_text)

# Set up chains
llm_chain = LLMChain(llm=llm_extract, prompt=extract_prompt)

combine_docs_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context"
)

qa_chain = RetrievalQA(
    retriever=retriever,
    combine_documents_chain=combine_docs_chain
)

# LangGraph node
def extract_node(state: MessagesState):
    query = state["messages"][-1].content
    print("[DEBUG] Extracting with query:", query)
    result = qa_chain.invoke({"query": query})
    print("\n[EXTRACTION RESULT]\n", result["result"])
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": result["result"]}]
    }

# Build LangGraph
builder = StateGraph(MessagesState)
builder.add_node("extract", extract_node)
builder.set_entry_point("extract")
builder.set_finish_point("extract")
graph = builder.compile()

# LangGraph Studio hook
def get_graph():
    return graph

# CLI Runner - Logic copied from your working file
if __name__ == "__main__":
    # Go up two levels to get project root from current script
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
     
    # Create output dir relative to root (e.g., your-project/data/output)
    output_dir = os.path.join(project_root, "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    # File path - Customized for this agent's output
    output_path = os.path.join(output_dir, "known_sites_lit_output.txt")

    # Invoke the graph and collect messages
    output_lines = []
    # User message - Customized for this agent's purpose
    user_message = {"role": "user", "content": "List all known archaeological sites in the Amazon described in the literature."}
    result = graph.invoke({"messages": [user_message]})
     
    for msg in result["messages"]:
        role = getattr(msg, "role", type(msg).__name__.replace("Message", "").lower())
        content = getattr(msg, "content", str(msg))
        print(f"{role.upper()}: {content}")
        output_lines.append(f"{role.upper()}: {content}")

    # Save the final output to a file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(output_lines))
     
    print(f"\n[SUCCESS] Saved output to {output_path}")
