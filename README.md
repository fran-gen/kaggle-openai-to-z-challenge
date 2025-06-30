# Kaggle OpenAI to Z-Challenge

This project is a multi-agent system designed to identify potential locations of undiscovered pre-Columbian archaeological sites in the Amazon rainforest.

---

## Description

The project uses a sequence of three distinct agents, each with a specific task, to first gather information on known archaeological sites and then use that data to predict the locations of unknown sites. It leverages **Large Language Models (LLMs)** and a **Retrieval-Augmented Generation (RAG)** architecture to process information from Wikipedia and a curated collection of scientific literature.

---

## Workflow

The system operates in a **three-stage pipeline**. The agents must be run in the specified order to ensure the final output is based on the complete set of gathered information.

1. **RAG Agent** (`agent.py`)  
   - This is the **first** agent.  
   - It synthesizes information from the other agents and queries the scientific literature to predict potential locations for new, undiscovered archaeological sites.  
   - Its findings are reviewed and refined by a second LLM for improved scientific rigor.

2. **Literature Agent** (`agent_literature.py`)  
   - This agent is run **second**.  
   - It queries a **FAISS vector store** (built from scientific papers) to extract information about other known sites mentioned within the literature.

3. **Wiki Agent** (`agent_wiki.py`)  
   - This agent is run **last**.  
   - It extracts structured information (name, coordinates, environmental features) about known archaeological sites from a hardcoded list of Wikipedia pages.  
   - Includes a review step to ensure data completeness.

---

## Installation

1. **Clone the repository:**

```bash
   git clone https://github.com/your-username/kaggle-openai-to-z-challenge.git
   cd kaggle-openai-to-z-challenge
````

2. **Create and activate a virtual environment:**

```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. **Install dependencies using Poetry:**

```bash
   pip install poetry
   poetry install
```

---

## Usage

Run the agents from the command line in the following order. Ensure you are in the **root directory** of the project.

### 1. Run the RAG Agent:

```bash
python -m src.agents.agent
```

* Output: `data/output/unknown_sites_lit_output.txt`

### 2. Run the Literature Agent:

```bash
python -m src.agents.agent_literature
```

* Output: `data/output/known_sites_lit_output.txt`

### 3. Run the Wiki Agent:

```bash
python -m src.agents.agent_wiki
```

* Output: `data/output/known_sites_wiki_output.txt`

---

## Dependencies

Project dependencies are managed with **Poetry** and are listed in the `pyproject.toml` file. Key libraries include:

* `langgraph`
* `langchain-openai`
* `python-dotenv`
* `faiss-cpu`

---

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.
