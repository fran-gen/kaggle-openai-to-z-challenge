import os
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, MessagesState, StateGraph

# Load environment variables
load_dotenv()

# Get the abs path to the project root (one level up from this script's directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Build the full path to the FAISS vectorstore directory inside the project
vectorstore_path = os.path.join(project_root, "vectorstore/faiss_index")

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

# Prompt for the initial RAG step
rag_prompt = PromptTemplate.from_template("""
You are a geoarchaeologist tasked with identifying regions in the Amazon rainforest likely to contain undiscovered pre-Columbian archaeological sites. Base your answer on the following context:

{context}
                                          
Back up your answer with scientific reasoning and relevant references from the context.                                        

Question: {question}
""")

# LLMs
llm_rag = ChatOpenAI(model="gpt-4.1", temperature=0.2)
llm_review = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Set up RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_rag,
    retriever=retriever,
    chain_type_kwargs={"prompt": rag_prompt},
)

# Node 1: Generate RAG answer using gpt-4.1
def rag_node(state: MessagesState):
    user_message = state["messages"][-1].content
    print("[DEBUG] RAG invoked with:", user_message)
    result = qa_chain.invoke({"query": user_message})
    answer_text = result["result"]
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": answer_text}]
    }

# Node 2: Refine the answer with context using gpt-4o-mini
def review_node(state: MessagesState):
    user_message = state["messages"][-2].content  # original user query
    original_answer = state["messages"][-1].content  # RAG output from node 1

    # Re-run retrieval for reviewer
    docs = retriever.get_relevant_documents(user_message)
    context = "\n\n".join([doc.page_content for doc in docs])

    review_prompt = f"""
You are a scientific expert in geoarchaeology. Your task is to refine and strengthen the following geoarchaeological answer using the retrieved context below.

Context:
{context}

Original Answer:
{original_answer}

Please reformulate the answer to improve scientific rigor and completeness. At the end, briefly explain the improvements you made.
"""

    print("[DEBUG] Reviewer evaluating answer with context...")
    response = llm_review.invoke(review_prompt)

    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response.content}]
    }

# Build the graph
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

# CLI runner
if __name__ == "__main__":
    
    # Go up twp levels to get project root from current script
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Create output dir relative to root (openai-to-z-challenge/data/output)
    output_dir = os.path.join(project_root, "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    # File path
    output_path = os.path.join(output_dir, "unknown_sites_output.txt")

    # Collect all assistant messages
    output_lines = []
    user_message = {"role": "user", "content": "Where might there be unknown archaeological sites in the Amazon?"}
    result = graph.invoke({"messages": [user_message]})
    for msg in result["messages"]:
        role = getattr(msg, "role", type(msg).__name__.replace("Message", "").lower())
        content = getattr(msg, "content", str(msg))
        print(f"{role.upper()}: {content}")
        output_lines.append(f"{role.upper()}: {content}")

    # Save combined assistant messages only
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(output_lines))
