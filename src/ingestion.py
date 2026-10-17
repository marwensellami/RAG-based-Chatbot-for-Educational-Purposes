"""
src/ingestion.py — Pipeline : Loading → Splitting → Indexing → Storing
VectorStore : FAISS (natif LlamaIndex, persistance unifiée)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import pdfplumber
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
    EMBED_DIM,
    EMBED_MODEL,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    INDEX_DIR,
    OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)
_SUPPORTED = {".pdf", ".md", ".txt", ".py", ".ipynb"}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _extract_pdf(path: Path) -> str:
    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            parts: list[str] = []
            raw = page.extract_text()
            if raw and raw.strip():
                parts.append(raw.strip())
            for table in page.extract_tables() or []:
                rows = [
                    " | ".join(str(c).strip() if c is not None else "" for c in row)
                    for row in table if row
                ]
                if rows:
                    parts.append("\n".join(rows))
            if parts:
                pages.append(f"[Page {i}]\n" + "\n\n".join(parts))
    return "\n\n".join(pages)


def _extract_notebook(path: Path) -> str:
    try:
        data  = json.loads(path.read_text(encoding="utf-8"))
        parts = []
        for cell in data.get("cells", []):
            source = "".join(cell.get("source", []))
            if not source.strip():
                continue
            if cell.get("cell_type") == "code":
                parts.append(f"```python\n{source}\n```")
            else:
                parts.append(source)
        return "\n\n".join(parts)
    except Exception as exc:
        logger.warning("Notebook illisible %s : %s", path.name, exc)
        return ""


def _extract_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


def step_loading(docs_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.is_dir() or path.suffix.lower() not in _SUPPORTED:
            continue
        try:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                text = _extract_pdf(path)
            elif suffix == ".ipynb":
                text = _extract_notebook(path)
            else:
                text = _extract_text(path)
        except Exception as exc:
            logger.error("Extraction échouée [%s] : %s", path.name, exc)
            continue

        if not text.strip():
            logger.warning("Fichier vide ignoré : %s", path.name)
            continue

        documents.append(Document(
            text=text,
            metadata={
                "file_name": path.name,
                "file_path": str(path),
                "file_type": path.suffix.lower(),
            },
        ))
        logger.info("[LOADING]   ✓ %s — %d caractères", path.name, len(text))

    logger.info("[LOADING]   %d document(s) chargé(s)", len(documents))
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SPLITTING
# ══════════════════════════════════════════════════════════════════════════════

def step_splitting(documents: list[Document]):
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        paragraph_separator="\n\n",
    )
    nodes = []
    for doc in documents:
        nodes.extend(splitter.get_nodes_from_documents([doc]))
    logger.info("[SPLITTING] %d chunks (size=%d overlap=%d)",
                len(nodes), CHUNK_SIZE, CHUNK_OVERLAP)
    return nodes


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3+4 — INDEXING + STORING avec FAISS
# ══════════════════════════════════════════════════════════════════════════════

def step_index_and_store(nodes) -> None:
    """
    Crée un index FAISS, indexe les nœuds et persiste tout
    en un seul appel storage_context.persist().
    FAISS + docstore + index_store sont sauvegardés ensemble.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Créer un index HNSW — plus rapide et plus précis que Flat pour >1k vecteurs
    faiss_index = faiss.IndexHNSWFlat(EMBED_DIM, HNSW_M)
    faiss_index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    faiss_index.hnsw.efSearch        = HNSW_EF_SEARCH
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    logger.info(
        "[HNSW]      M=%d | efConstruction=%d | efSearch=%d",
        HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH,
    )

    storage_ctx = StorageContext.from_defaults(
        vector_store=vector_store,
    )

    logger.info("[INDEXING]  Indexation de %d nœuds…", len(nodes))
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_ctx,
        show_progress=True,
    )

    # Persister TOUT en un seul appel — FAISS + docstore + index_store
    storage_ctx.persist(persist_dir=str(INDEX_DIR))
    logger.info("[STORING]   Index FAISS persisté dans %s", INDEX_DIR)

    # Vérifier la taille du docstore
    docstore_path = INDEX_DIR / "docstore.json"
    if docstore_path.exists():
        logger.info("[STORING]   docstore.json : %d octets",
                    docstore_path.stat().st_size)


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrateur
# ══════════════════════════════════════════════════════════════════════════════

def run_ingestion(force: bool = False) -> None:
    """Loading → Splitting → Indexing+Storing (FAISS)"""
    docs_dir = Path(DOCS_DIR)
    if not docs_dir.exists():
        logger.error("Dossier docs introuvable : %s", docs_dir)
        return

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    logger.info("=" * 52)
    logger.info("PIPELINE INGESTION — démarrage (FAISS)")
    logger.info("=" * 52)

    documents = step_loading(docs_dir)
    if not documents:
        logger.error("Aucun document. Pipeline arrêté.")
        return

    nodes = step_splitting(documents)
    if not nodes:
        logger.error("Aucun chunk. Pipeline arrêté.")
        return

    step_index_and_store(nodes)

    logger.info("=" * 52)
    logger.info("PIPELINE INGESTION — terminé avec succès")
    logger.info("=" * 52)


def run_ingestion_if_needed() -> None:
    faiss_path    = INDEX_DIR / "default__vector_store.json"
    docstore_path = INDEX_DIR / "docstore.json"
    needs_run = (
        not faiss_path.exists()
        or not docstore_path.exists()
        or docstore_path.stat().st_size < 512
    )
    if needs_run:
        logger.info("Index absent ou vide — ingestion lancée…")
        run_ingestion()
    else:
        logger.info("Index FAISS OK — skip ingestion.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    run_ingestion(force=True)

