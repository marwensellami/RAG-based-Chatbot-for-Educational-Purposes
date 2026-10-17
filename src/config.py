"""
src/config.py — Configuration centralisée PyTutor
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Répertoires ────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
DOCS_DIR  = BASE_DIR / "data" / "docs"
INDEX_DIR = BASE_DIR / "data" / "index"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL       = os.getenv("LLM_MODEL",       "qwen2.5:7b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
EMBED_MODEL     = os.getenv("EMBED_MODEL",     "nomic-embed-text-v2-moe")

# ── FAISS — dimension des vecteurs ─────────────────────────────────────────────
# nomic-embed-text        = 768
# nomic-embed-text-v2-moe = 768
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# ── FAISS HNSW ────────────────────────────────────────────────────────────────
# M              : connexions par nœud (16–64). Plus élevé = meilleur rappel,
#                  plus de RAM. 32 est un bon compromis général.
# EF_CONSTRUCTION: taille de la liste dynamique pendant la construction.
#                  Plus élevé = index de meilleure qualité, build plus lent.
# EF_SEARCH      : taille de la liste dynamique pendant la recherche.
#                  Plus élevé = meilleur rappel, recherche un peu plus lente.
HNSW_M               = int(os.getenv("HNSW_M",               "32"))
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
HNSW_EF_SEARCH       = int(os.getenv("HNSW_EF_SEARCH",       "64"))

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "256"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "30"))

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K             = int(os.getenv("TOP_K",             "4"))
SIMILARITY_CUTOFF = float(os.getenv("SIMILARITY_CUTOFF", "0.1"))

# ── Mémoire ────────────────────────────────────────────────────────────────────
MEMORY_TOKEN_LIMIT = int(os.getenv("MEMORY_TOKEN_LIMIT", "999999"))

