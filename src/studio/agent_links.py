import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, MessagesState

# Load environment variables
load_dotenv()

# LLMs
llm_extract = ChatOpenAI(model="gpt-4.1", max_tokens=4096, temperature=0.2)
llm_review = ChatOpenAI(model="gpt-4o-mini", max_tokens=4096, temperature=0)

# Prompt: extract site info from Wikipedia page
extract_prompt = PromptTemplate.from_template("""
You are a scientific assistant helping to extract structured information from Wikipedia pages listing archaeological sites in the Amazonas Region:

1. https://en.wikipedia.org/wiki/Caraj%C3%ADa
2. https://en.wikipedia.org/wiki/Gran_Saposoa
3. https://en.wikipedia.org/wiki/Gran_Vilaya
4. https://en.wikipedia.org/wiki/Ku%C3%A9lap
5. https://en.wikipedia.org/wiki/Laguna_de_las_Momias
6. https://en.wikipedia.org/wiki/Llactan
7. https://en.wikipedia.org/wiki/Machu_Pirqa                                              
8. https://en.wikipedia.org/wiki/Purunllacta_(Cheto)
9. https://en.wikipedia.org/wiki/Purunllacta,_Soloco                                              
10. https://en.wikipedia.org/wiki/Carachupa    
11. https://en.wikipedia.org/wiki/Cochabamba_(archaeological_site) 
12. https://en.wikipedia.org/wiki/Revash
13. https://en.wikipedia.org/wiki/Sarcophagi_of_Caraj%C3%ADa                                                                                      
14. https://en.wikipedia.org/wiki/Wilca
                                              
Your task is to extract for each known site:
- The site's name
- The site's geographic coordinates 
- The country it is located in (e.g., Peru, Brazil, Colombia)
- A list of key geographic or environmental features useful to identify or describe the site
  (e.g., elevation, vegetation type, river proximity, soil, mineral composition, climate)
- Be exhaustive and include all sites mentioned in the main page and any linked subcategories. Take your time to ensure completeness.

Return a Python list of dictionaries like this:

[
  {{
    "site": "Name of site",
    "coordinates": "DMS",
    "features": "brief description of features"
  }},
  ...
]
""")

# Extraction node

def extract_node(state: MessagesState):
    print("[DEBUG] Running site extraction node...")
    message = extract_prompt.format()
    response = llm_extract.invoke(message)
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response.content}],
        "counter": state.get("counter", 0) + 1
    }

# Review node with logic to loop back if incomplete

def review_node(state: MessagesState):
    last_response = state["messages"][-1].content
    counter = state.get("counter", 1)

    review_prompt = f"""
Review the following list of archaeological sites. Ensure that **every** entry includes:
- Site name
- Coordinates (in DMS format, e.g. "6°22′S 90°52′W")
- Country
- Environmental features (non-empty string)

If any of these are missing or incomplete in **any** entry, respond with:
REVISE

Otherwise, return:
COMPLETE
"""
    print("[DEBUG] Reviewing for completeness...")
    review = llm_review.invoke(review_prompt + f"\n\n{last_response}")

    if "REVISE" in review.content and counter < 10:
        print("[DEBUG] Incomplete — rerunning extraction")
        return {"messages": state["messages"], "counter": counter}  # keep counter and loop back
    else:
        print("[DEBUG] Review complete or max retries reached")
        return {"messages": state["messages"] + [{"role": "assistant", "content": last_response}]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("extract", extract_node)
builder.add_node("review", review_node)

builder.set_entry_point("extract")
builder.add_conditional_edges(
    "review",
    lambda state: "extract" if state.get("counter", 0) < 10 and "REVISE" in state["messages"][-1].content else END
)
builder.add_edge("extract", "review")

graph = builder.compile()

# LangGraph Studio hook
def get_graph():
    return graph

# CLI runner
if __name__ == "__main__":
    user_message = {
        "role": "user",
        "content": "Extract all known archaeological sites in the Amazonas region from Wikipedia, with coordinates and features. Be exhaustive"
    }
    result = graph.invoke({"messages": [user_message], "counter": 0})

    # Go up three levels to get project root from current script
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Create output dir relative to root (openai-to-z-challenge/data/output)
    output_dir = os.path.join(project_root, "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    # File path
    output_path = os.path.join(output_dir, "sites_output.txt")

    # Collect all assistant messages
    output_lines = []
    for msg in result["messages"]:
        role = getattr(msg, "role", type(msg).__name__.replace("Message", "").lower())
        content = getattr(msg, "content", str(msg))
        print(f"{role.upper()}: {content}")
        if role == "assistant":
            output_lines.append(content)

    # Save combined assistant messages only
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(output_lines))
