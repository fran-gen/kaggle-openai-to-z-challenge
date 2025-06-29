import os
from dotenv import load_dotenv
from typing import TypedDict, List
# NEW: Import the message classes to fix the error
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Load environment variables
load_dotenv()

# Helper function to load prompts from files
def load_prompt(file_name: str) -> str:
    """Loads a prompt template from a file."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    prompt_path = os.path.join(project_root, "src", "prompts", file_name)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_path}")
        raise

# Define a more specific state for our graph
class GraphState(TypedDict):
    messages: List[BaseMessage]
    counter: int
    review_decision: str  # "REVISE" or "COMPLETE"

# LLMs
llm_extract = ChatOpenAI(model="gpt-4-turbo", max_tokens=4096, temperature=0.2)
llm_review = ChatOpenAI(model="gpt-4o-mini", max_tokens=4096, temperature=0)

# Load prompts from .txt files
extract_prompt_text = load_prompt("extract_known_sites_wiki_prompt.txt")
review_prompt_text = load_prompt("review_known_sites_wiki_prompt.txt")

# Create the PromptTemplate object from the loaded text
extract_prompt = PromptTemplate.from_template(extract_prompt_text)


# Graph Nodes
def extract_node(state: GraphState):
    """Invokes the extraction LLM and increments the attempt counter."""
    print(f"[DEBUG] Running site extraction node (Attempt #{state.get('counter', 0) + 1})...")
    message = extract_prompt.format()
    response = llm_extract.invoke(message)
    # CHANGED: Append a proper AIMessage object, not a dict, to the state.
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "counter": state.get("counter", 0) + 1,
    }


def review_node(state: GraphState):
    """Invokes the review LLM and saves its decision to the state."""
    print("[DEBUG] Reviewing for completeness...")
    # This line will now work correctly because the last message is an object.
    last_response = state["messages"][-1].content
    counter = state.get("counter", 0)

    full_review_prompt = review_prompt_text + f"\n\nHere is the list to review:\n{last_response}"
    review = llm_review.invoke(full_review_prompt)

    if "REVISE" in review.content and counter < 5:
        print(f"[DEBUG] Review result: REVISE.")
        decision = "REVISE"
    else:
        if "REVISE" in review.content:
            print("[DEBUG] Max retries reached. Forcing completion.")
        else:
            print("[DEBUG] Review result: COMPLETE.")
        decision = "COMPLETE"

    return {"review_decision": decision}


def should_continue(state: GraphState):
    """The router for our conditional edge, based on the review node's decision."""
    if state["review_decision"] == "REVISE":
        return "extract"
    else:
        return END


# Build graph
builder = StateGraph(GraphState)
builder.add_node("extract", extract_node)
builder.add_node("review", review_node)
builder.set_entry_point("extract")
builder.add_edge("extract", "review")
builder.add_conditional_edges(
    "review",
    should_continue,
    {
        "extract": "extract",
        END: END,
    },
)

graph = builder.compile()


# CLI Runner
if __name__ == "__main__":
    # CHANGED: Create a proper HumanMessage object for the initial state.
    user_message = HumanMessage(
        content="Extract all known archaeological sites in the Amazonas region from Wikipedia, "
        "with coordinates and features. Be exhaustive in the description provided. Do not leave fields not specified.",
    )

    initial_state = {
        "messages": [user_message],
        "counter": 0,
        "review_decision": "",
    }

    result = graph.invoke(initial_state)

    # File Output Logic
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = os.path.join(project_root, "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "known_sites_wiki_output.txt")

    # The final, reviewed message is the last one in the list.
    final_assistant_message = result["messages"][-1].content
    
    print("\n--- FINAL, REVIEWED OUTPUT ---")
    print(final_assistant_message)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_assistant_message)

    print(f"\n[SUCCESS] Saved final reviewed output to: {output_path}")
