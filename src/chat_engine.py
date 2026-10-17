"""
src/chat_engine.py — Orchestration LlamaIndex — PyTutor
VectorStore : FAISS (natif LlamaIndex)
"""
from __future__ import annotations

import logging
from pathlib import Path

# Silence le warning "BM25Retriever does not support embeddings"
# Ce warning vient de LlamaIndex lui-même, notre code est correct
logging.getLogger("llama_index.retrievers.bm25.base").setLevel(logging.ERROR)

import faiss
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.faiss import FaissVectorStore

from src.config import (
    EMBED_DIM,
    EMBED_MODEL,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    INDEX_DIR,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MEMORY_TOKEN_LIMIT,
    OLLAMA_BASE_URL,
    SIMILARITY_CUTOFF,
    TOP_K,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Tu es PyTutor, un assistant pédagogique spécialisé en Python.
Ton rôle est d'expliquer les concepts Python de manière claire, progressive
et adaptée au niveau de l'étudiant.

Règles :
- Illustre toujours avec des exemples de code Python concrets et commentés.
- Adapte le niveau d'explication à la question (débutant → expert).
- Décompose les notions complexes en étapes simples.
- Cite les sources/extraits pertinents quand disponibles.
- Pour les questions hors Python, oriente poliment vers le sujet.
- Réponds en français sauf si l'utilisateur écrit en anglais.
"""

_CONDENSE_PROMPT = (
    "Historique de la conversation :\n"
    "{chat_history}\n\n"
    "Question de suivi : {question}\n\n"
    "Reformule cette question en une question autonome et complète, "
    "compréhensible sans l'historique.\n"
    "Question reformulée : "
)

_CONTEXT_PROMPT = (
    "Extraits pertinents de la documentation Python "
    "(recherche hybride sémantique + mots-clés) :\n"
    "{context_str}\n\n"
    "Utilise ces informations pour enrichir et préciser ta réponse."
)


def _load_index():
    faiss_path = Path(INDEX_DIR) / "default__vector_store.json"
    if not faiss_path.exists():
        logger.warning("Index HNSW introuvable. Lancer : python -m src.ingestion")
        faiss_index = faiss.IndexHNSWFlat(EMBED_DIM, HNSW_M)
        faiss_index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        faiss_index.hnsw.efSearch        = HNSW_EF_SEARCH
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_ctx  = StorageContext.from_defaults(vector_store=vector_store)
        from llama_index.core import VectorStoreIndex
        return VectorStoreIndex(nodes=[], storage_context=storage_ctx)

    vector_store = FaissVectorStore.from_persist_dir(str(INDEX_DIR))
    storage_ctx  = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=str(INDEX_DIR),
    )
    index = load_index_from_storage(storage_context=storage_ctx)
    logger.info("Index HNSW chargé — %d nœuds (M=%d efSearch=%d)",
                len(index.docstore.docs), HNSW_M, HNSW_EF_SEARCH)
    return index


def _build_retriever(index):
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=TOP_K)

    nodes = list(index.docstore.docs.values())
    if not nodes:
        logger.warning("Docstore vide — BM25 désactivé.")
        return vector_retriever

    # BM25 vérifie seulement nodes[0].embedding pour afficher le warning
    # Il suffit de le mettre à None sur tous les nœuds (pas besoin de deepcopy)
    for node in nodes:
        node.embedding = None

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=TOP_K,
    )

    fusion = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=TOP_K,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )
    logger.info("Retriever hybride — FAISS + BM25 + RRF (TOP_K=%d, %d nœuds)", TOP_K, len(nodes))
    return fusion


def build_chat_engine() -> CondensePlusContextChatEngine:
    # 1. LLM
    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=900.0,
        system_prompt=_SYSTEM_PROMPT,
        temperature=LLM_TEMPERATURE,
    )

    # 2. Embeddings
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    logger.info("Embeddings : %s", EMBED_MODEL)

    # 3. Index
    index = _load_index()

    # 4. Retriever
    retriever = _build_retriever(index)

    # 5. ChatEngine
    engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        memory=ChatMemoryBuffer.from_defaults(token_limit=MEMORY_TOKEN_LIMIT),
        llm=Settings.llm,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)],
        context_prompt=_CONTEXT_PROMPT,
        condense_prompt=_CONDENSE_PROMPT,
        verbose=True,
    )

    logger.info("ChatEngine prêt — %s | HNSW | RRF", LLM_MODEL)
    return engine

