# 🎓 PyTutor — RAG-Based Chatbot for Educational Purposes

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-purple.svg)](https://www.llamaindex.ai/)
[![Ollama](https://img.shields.io/badge/Ollama-Local-black.svg)](https://ollama.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange.svg)](https://github.com/facebookresearch/faiss)
[![Gradio](https://img.shields.io/badge/Gradio-Web_UI-ff69b4.svg)](https://www.gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**PyTutor** is a Retrieval-Augmented Generation (RAG) chatbot designed for educational purposes. It answers questions about Python courses and reference documentation by grounding its responses in source PDF documents. The application runs locally using Ollama, LlamaIndex, and FAISS, without requiring external LLM APIs.

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#️-configuration)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [License](#-license)
- [Author](#-author)
- [Acknowledgements](#-acknowledgements)

---

## 🧭 Overview

PyTutor implements a complete Retrieval-Augmented Generation pipeline:

1. **Ingestion:** PDF documents, such as `cours-python.pdf` and `docs.pdf`, are parsed and divided into overlapping chunks. Embeddings are generated and stored in a FAISS HNSW index.
2. **Retrieval:** When a user asks a question, the system retrieves relevant passages using semantic similarity search.
3. **Generation:** A local LLM, powered by Ollama and using `qwen2.5:7b` by default, generates an answer based on the retrieved context.
4. **Conversation:** A Gradio web interface provides an interactive chat experience with conversational context.

The application is designed to run locally without requiring an OpenAI API key or a cloud-based LLM service. Internet access may still be needed to download models and install dependencies during initial setup.

---

## ✨ Key Features

- 🔒 **Local inference:** LLM and embedding models run through Ollama.
- 📚 **Document-grounded answers:** Responses are generated using passages retrieved from the source PDFs.
- ⚡ **Efficient vector search:** FAISS HNSW indexing supports approximate nearest-neighbor retrieval.
- 🧠 **Conversational context:** Supports multi-turn conversations.
- 🎛️ **Configurable pipeline:** Model selection, chunking, retrieval parameters, and similarity thresholds can be configured through environment variables.
- 🖥️ **Interactive web interface:** A Gradio interface provides a simple chat experience without requiring a separate frontend.
- 🐍 **Educational focus:** Helps students understand Python using their own course materials.
- 🔐 **Data privacy:** Documents and queries can remain on the local machine when all services are configured locally.

---

## 🏗️ Architecture

```text
┌─────────────────────────┐
│     PDF Documents       │
│       data/docs/        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   PDF Parsing           │
│   Chunking              │
│   Text Preprocessing    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Embedding Generation  │
│   Ollama                │
│   nomic-embed-text-v2-moe│
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     FAISS HNSW Index    │
│       data/index/       │
└────────────┬────────────┘
             │
             │ Retrieval (Top-K)
             ▼
┌─────────────────────────┐
│    Context Retrieval    │
│    + Prompt Builder     │
└────────────▲────────────┘
             │
┌────────────┴────────────┐
│      User Question      │
└─────────────────────────┘
             │
             ▼
┌─────────────────────────┐
│    Ollama LLM           │
│    Qwen2.5 7B           │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     Gradio Chat UI      │
└─────────────────────────┘
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Programming language | Python 3.11+ |
| User interface | Gradio |
| RAG orchestration | LlamaIndex |
| Language model | Ollama with `qwen2.5:7b` |
| Embeddings | Ollama with `nomic-embed-text-v2-moe` |
| Embedding dimension | 768, subject to model configuration |
| Vector store | FAISS HNSW |
| PDF parsing | pypdf, pdfplumber, pdfminer.six |
| Configuration | python-dotenv |

---

## 📁 Project Structure

```text
RAG-based-Chatbot-for-Educational-Purposes/
├── src/
│   ├── config.py          # Centralized configuration and .env loader
│   ├── ingestion.py       # PDF parsing and index building
│   └── chat_engine.py     # RAG engine: retrieval and generation
├── data/
│   ├── docs/              # Source PDF documents
│   └── index/             # Generated FAISS index and metadata
├── app.py                 # Gradio application entry point
├── requirements.txt       # Python dependencies
├── .env.example           # Example environment configuration
├── .gitignore             # Files excluded from version control
└── README.md              # Project documentation
```

The `data/index/`, virtual environment directories, `.env`, and `.gradio/` directories should be excluded from version control where appropriate.

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or later
- [Ollama](https://ollama.com/) installed and running
- Git
- Sufficient RAM and disk space for the selected LLM and embedding models

### 1. Clone the Repository

```bash
git clone https://github.com/marwensellami/RAG-based-Chatbot-for-Educational-Purposes.git
cd RAG-based-Chatbot-for-Educational-Purposes
```

### 2. Create and Activate a Virtual Environment

Create the environment:

```bash
python -m venv venv
```

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```bat
venv\Scripts\activate.bat
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Pull the Ollama Models

Download the default language model:

```bash
ollama pull qwen2.5:7b
```

Download the embedding model:

```bash
ollama pull nomic-embed-text-v2-moe
```

Make sure Ollama is running before launching the application.

### 5. Configure the Environment

Create a local environment file.

**Windows PowerShell:**

```powershell
Copy-Item .env.example .env
```

**Linux/macOS:**

```bash
cp .env.example .env
```

Edit `.env` if you need to change the model names, Ollama server URL, or retrieval parameters.

### 6. Add the Source Documents

Place the PDF documents in `data/docs/`.

For example:

```text
data/docs/
├── cours-python.pdf
└── docs.pdf
```

Use your actual course and reference PDFs. These files are not necessarily included in the repository.

### 7. Build the Vector Index

Run the ingestion module from the project root:

```bash
python -m src.ingestion
```

This step parses the PDFs, generates embeddings, and creates the FAISS index and associated metadata in `data/index/`.

Run the ingestion process again whenever the source documents change.

### 8. Launch the Application

From the project root, execute:

```bash
python app.py
```

The Gradio interface is expected to be available at:

http://127.0.0.1:7860

The actual address and port depend on the Gradio configuration in `app.py`.

---

## ⚙️ Configuration

The application configuration is managed through `src/config.py`, with optional overrides provided by `.env`.

The following table describes the configuration variables used by the documented setup. Ensure that their names and default values match your actual implementation.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `qwen2.5:7b` | Language model used for generation |
| `LLM_TEMPERATURE` | `0.2` | Controls generation randomness |
| `EMBED_MODEL` | `nomic-embed-text-v2-moe` | Embedding model |
| `EMBED_DIM` | `768` | Expected embedding dimension |
| `HNSW_M` | `32` | Number of HNSW graph connections |
| `HNSW_EF_CONSTRUCTION` | `200` | Search breadth during index construction |
| `HNSW_EF_SEARCH` | `64` | Search breadth during retrieval |
| `CHUNK_SIZE` | `256` | Chunk size; unit depends on the implementation |
| `CHUNK_OVERLAP` | `30` | Overlap between consecutive chunks |
| `TOP_K` | `4` | Maximum number of retrieved passages |
| `SIMILARITY_CUTOFF` | `0.1` | Minimum score threshold for retaining passages |
| `MEMORY_TOKEN_LIMIT` | `999999` | Configured conversation-memory limit |

**Important:** Verify the embedding dimension and similarity-score interpretation against your actual embedding model and retrieval implementation. The meaning of `SIMILARITY_CUTOFF` depends on how the application calculates and filters scores.

---

## 🎮 Usage

Once the application is running:

1. Open the Gradio URL displayed in the terminal.
2. Enter a question about the content of your Python course.
3. Submit your question and wait for the retrieved context and generated answer.
4. Continue the conversation to ask follow-up questions using the available conversational context.

Example questions:

- *What is a list comprehension in Python?*
- *Explain the difference between a tuple and a list.*
- *How does exception handling work in Python?*
- *What are the main differences between a list, a set, and a dictionary?*

The chatbot retrieves relevant passages from the indexed documents and uses them as context for generating its response.

---

## 🔬 How It Works

### 1. Document Ingestion — `src/ingestion.py`

The ingestion module prepares the knowledge base:

- Loads PDF files from `data/docs/`.
- Extracts text using the configured PDF parsing libraries.
- Splits the extracted text into chunks.
- Generates embeddings using the configured Ollama embedding model.
- Builds and persists the FAISS HNSW index and associated metadata.

### 2. Retrieval and Generation — `src/chat_engine.py`

The chat engine handles user questions:

- Receives the user's question.
- Converts the question into an embedding.
- Searches the FAISS index for relevant passages.
- Applies the configured retrieval and similarity-filtering parameters.
- Builds a prompt from the retrieved context and conversation history.
- Sends the prompt to the local Ollama language model.
- Returns the generated answer to the interface.

### 3. User Interface — `app.py`

The Gradio application provides the interaction layer:

- Displays the chat interface.
- Accepts user questions.
- Presents generated answers.
- Maintains conversational context according to the application's memory implementation.
- Loads the retrieval components when the application starts, depending on the implementation.

---

## 🛠️ Troubleshooting

| Issue | Possible Solution |
|---|---|
| Connection refused by Ollama | Start Ollama and verify that `OLLAMA_BASE_URL` points to the correct server. |
| Model not found | Run `ollama pull qwen2.5:7b` and `ollama pull nomic-embed-text-v2-moe`. |
| Empty or irrelevant answers | Verify that the PDFs contain extractable text, rebuild the index, and adjust retrieval parameters. |
| FAISS dimension mismatch | Ensure the query embeddings and indexed document embeddings have the same dimension. Rebuild the index after changing the embedding model. |
| Slow first response | The language model may need to load into memory. Check available RAM and model size. |
| Gradio port already in use | Configure a different server port in `app.py`, for example `server_port=7861`. |
| PDF extraction problems | Check whether the PDF contains selectable text or requires OCR. |
| Missing Python dependencies | Activate the correct virtual environment and reinstall the packages from `requirements.txt`. |

---

## 🗺️ Roadmap

- [ ] Add BM25 hybrid retrieval to combine keyword search and vector similarity.
- [ ] Support multiple document collections with per-document filtering.
- [ ] Add evaluation metrics such as faithfulness, answer relevancy, and retrieval precision.
- [ ] Provide a Docker Compose configuration for the application and its dependencies.
- [ ] Add conversation transcript export and import.
- [ ] Improve source attribution by displaying document names and relevant page numbers.
- [ ] Add automated tests for document ingestion, retrieval, and answer generation.

---

## 📄 License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

**Document copyright notice:** The PDF documents placed in `data/docs/` may be subject to separate copyright restrictions. Ensure you have permission to redistribute them before including them in a public repository.

---

## 👤 Author

**Marwen Sellami**

- GitHub: [@marwensellami](https://github.com/marwensellami)
- Repository: [PyTutor — RAG-Based Chatbot for Educational Purposes](https://github.com/marwensellami/RAG-based-Chatbot-for-Educational-Purposes)

---

## 🙏 Acknowledgements

- [LlamaIndex](https://www.llamaindex.ai/) — RAG orchestration framework.
- [Ollama](https://ollama.com/) — Local LLM and embedding model runtime.
- [FAISS](https://github.com/facebookresearch/faiss) — Efficient vector similarity search.
- [Gradio](https://www.gradio.app/) — Interactive web interface.
- [pypdf](https://pypdf.readthedocs.io/) — PDF parsing.
- The open-source community for the tools and libraries that make local RAG applications possible.

---

⭐ If you find this project useful for learning or experimentation, consider giving the repository a star!

**Disclaimer:** PyTutor is an educational project intended to support learning. Its generated answers may be incomplete or inaccurate, so users should verify important information against the original course materials and official documentation.
