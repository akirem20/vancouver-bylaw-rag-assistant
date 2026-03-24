# Vancouver Municipal Bylaw RAG Assistant

A Retrieval-Augmented Generation (RAG) pipeline designed for deterministic, grounded legal query processing.

## 🎯 Research Objective
To evaluate the efficacy of Small Language Models (SLMs) in embedding high-density legal text and mitigating "hallucination" in LLMs through semantic context injection. This project focuses on the City of Vancouver Street and Traffic By-law (No. 2849) .

## 🛠️ Technical Architecture
LLM Engine: Google Gemini 2.5 Flash

Embedding Model: all-MiniLM-L6-v2 (Sentence-Transformers) | 384-dimensional vector mapping.

Vector Store: ChromaDB (Persistent Client) for local, low-latency indexing.

Extraction: Automated PDF serialization via pypdf.

## 🚀 Key Features
Semantic Retrieval: Uses K-Nearest Neighbors (k=2) to identify relevant legal clauses based on intent rather than keywords.

Groundedness Verification: The system acknowledges context limitations to prevent out-of-distribution hallucinations (eg, identifying that passport regulations are federal, not municipal).



## 🛠️ Installation & Setup
1. Install Dependencies
Run the following command to install all necessary libraries:

pip install google-genai chromadb sentence-transformers pypdf python-dotenv

3. Environment Configuration
Create a .envfile in the root directory and add your API key:

GEMINI_API_KEY=your_actual_key_here

3. Run the Assistant
Ensure vancouver_law.pdfis in the root directory, then execute:

python main.py
