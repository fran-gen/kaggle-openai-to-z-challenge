import os
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
    # Assumes this script is in 'src/agents', so we go up two levels to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    prompt_path = os.path.join(project_root, "src", "prompts", file_name)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_path}")
        raise

# Path and Vectorstore Setup
# Project root relative to this script's location (e.g., /path/to/your-project/src)
project_src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Vectorstore path (e.g., /path/to/your-project/vectorstore/faiss_index)
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

# Load Prompts and Setup Chains (Corrected Method)
# Load prompt templates from files
rag_prompt_text = load_prompt("rag_agent_prompt.txt")
review_prompt_text = load_prompt("review_rag_agent_prompt.txt")

# Create a PromptTemplate object from the loaded text
rag_prompt = PromptTemplate.from_template(rag_prompt_text)

# LLMs
llm_rag = ChatOpenAI(model="gpt-4.1", temperature=0.2)
llm_review = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create the base LLMChain
# This chain combines the prompt and the LLM to answer a question based on context.
llm_chain = LLMChain(llm=llm_rag, prompt=rag_prompt)

# Create the StuffDocumentsChain
# This chain takes a list of documents, "stuffs" them into the {context} variable,
# and then runs the llm_chain.
combine_docs_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context"  # Explicitly name the context variable from the prompt
)

# Create the final RetrievalQA chain
# This top-level chain links the retriever and the document combination chain.
qa_chain = RetrievalQA(
    retriever=retriever,
    combine_documents_chain=combine_docs_chain
)

# Graph Nodes
# Node 1: Generate RAG answer using gpt-4.1
def rag_node(state: MessagesState):
    user_message = state["messages"][-1].content
    print("[DEBUG] RAG invoked with:", user_message)
    # The 'query' key is the standard input for RetrievalQA chains
    result = qa_chain.invoke({"query": user_message})
    answer_text = result["result"]
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": answer_text}]
    }

# Node 2: Refine the answer with context using gpt-4o-mini
def review_node(state: MessagesState):
    user_message = state["messages"][-2].content  # original user query
    original_answer = state["messages"][-1].content  # RAG output from node 1

    # Re-run retrieval for reviewer to get the most relevant context
    docs = retriever.get_relevant_documents(user_message)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Format the review prompt using the loaded template text
    final_review_prompt = review_prompt_text.format(
        context=context,
        original_answer=original_answer
    )

    print("[DEBUG] Reviewer evaluating answer with context...")
    response = llm_review.invoke(final_review_prompt)

    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response.content}]
    }

# Build and Compile Graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", rag_node)
builder.add_node("reviewer", review_node)
builder.set_entry_point("assistant")
builder.add_edge("assistant", "reviewer")
builder.set_finish_point("reviewer")

graph = builder.compile()

# LangGraph Studio hook
def get_graph():
    return graph

# CLI Runner
if __name__ == "__main__":
    # Go up two levels to get project root from current script
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Create output dir relative to root (e.g., your-project/data/output)
    output_dir = os.path.join(project_root, "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    # File path
    output_path = os.path.join(output_dir, "unknown_sites_lit_output.txt")

    # Invoke the graph and collect messages
    output_lines = []
    user_message = {"role": "user", "content": "Where might there be unknown archaeological sites in the Amazon? Base your answer on the latest research and known sites."}
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