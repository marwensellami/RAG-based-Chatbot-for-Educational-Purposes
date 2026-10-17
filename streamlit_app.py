"""
streamlit_app.py — Interface Streamlit — PyTutor
Streaming : message utilisateur affiché immédiatement + réponse progressive
"""
from __future__ import annotations

import logging
import shutil
import os
import time
from pathlib import Path

import streamlit as st
import requests

from src.chat_engine import build_chat_engine
from src.config import LLM_MODEL, OLLAMA_BASE_URL
from src.ingestion import run_ingestion, run_ingestion_if_needed

# ── Constantes ────────────────────────────────────────────────────────────────
OUT_OF_SCOPE_REPLY = (
    "⚠️ **Question hors périmètre**\n\n"
    "Je suis **PyTutor**, un assistant spécialisé exclusivement en **Python**.\n\n"
    "Je peux vous aider sur :\n"
    "- 🐍 Le langage Python et sa syntaxe\n"
    "- 📦 Les bibliothèques : NumPy, Pandas, Django, FastAPI, etc.\n"
    "- 🛠️ Les outils : pip, venv, pytest, mypy, ruff…\n"
    "- 💡 Les bonnes pratiques, algorithmes et design patterns Python\n"
    "- 🐛 Le débogage et les erreurs Python\n\n"
    "Reformulez votre question en lien avec Python et je serai ravi de vous aider ! 🐍"
)

_CLASSIFIER_MODEL   = os.getenv("CLASSIFIER_MODEL", "qwen2.5:0.5b")
_CLASSIFIER_RETRIES = 3
_CLASSIFIER_TIMEOUT = 60

_CLASSIFIER_SYSTEM = """\
Tu es un classificateur binaire strict.
Réponds UNIQUEMENT par OUI si la question porte sur Python (langage, bibliothèques, outils),
ou NON dans tous les autres cas. Aucune explication. Juste OUI ou NON.
"""

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="PyTutor - Assistant Python",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Header personnalisé */
    .pytutor-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f2040 100%);
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 20px;
        border: 1px solid #2d5a9e;
    }
    .pytutor-header h1 {
        margin: 0 0 6px 0;
        font-size: 1.8rem;
        color: #e8f4ff;
    }
    .pytutor-header p {
        margin: 0;
        color: #94b8d8;
        font-size: 0.92rem;
    }
    .model-badge {
        display: inline-block;
        background: rgba(15,32,64,0.8);
        border: 1px solid #2d5a9e;
        border-radius: 20px;
        padding: 3px 14px;
        font-size: 0.78rem;
        color: #64b5f6;
        margin-top: 10px;
        font-family: monospace;
    }
    .source-box {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 12px;
        font-size: 0.83rem;
        border-left: 3px solid #64b5f6;
    }
    .stChatMessage {
        padding: 0.5rem;
    }
    footer {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Initialisation ────────────────────────────────────────────────────────────
@st.cache_resource
def init_engine():
    """Initialise le moteur RAG (mis en cache pour une seule exécution)"""
    run_ingestion_if_needed()
    return build_chat_engine()

# ── Fonctions helpers ─────────────────────────────────────────────────────────
def _call_classifier(message: str, model: str) -> str | None:
    """Appelle le classificateur Ollama"""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model, "stream": False,
                "options": {"temperature": 0.0, "num_predict": 5},
                "messages": [
                    {"role": "system", "content": _CLASSIFIER_SYSTEM},
                    {"role": "user",   "content": message},
                ],
            },
            timeout=_CLASSIFIER_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip().upper()
    except Exception as exc:
        logger.warning("Classificateur [%s] : %s", model, exc)
    return None


def is_python_question(message: str) -> bool:
    """Vérifie si la question concerne Python"""
    models = list(dict.fromkeys([_CLASSIFIER_MODEL, LLM_MODEL]))
    for model in models:
        for attempt in range(1, _CLASSIFIER_RETRIES + 1):
            answer = _call_classifier(message, model)
            if answer is not None:
                is_python = answer.startswith("OUI")
                logger.info("Classificateur [%s] → %s", model, "Python" if is_python else "Hors-sujet")
                return is_python
    logger.error("Classificateur indisponible — fail-open.")
    return True


def format_sources(nodes) -> str:
    """Formate les sources pour l'affichage"""
    if not nodes:
        return "_Aucune source récupérée._"
    lines = ["**📚 Sources utilisées**\n"]
    for i, node in enumerate(nodes, 1):
        meta = node.metadata or {}
        fname = meta.get("file_name", "source inconnue")
        score = f"{node.score:.3f}" if hasattr(node, "score") and node.score else "—"
        snippet = node.get_content()[:200].replace("\n", " ").strip()
        file_type = meta.get("file_type", "")
        icon = "📄" if file_type == ".pdf" else "💻" if file_type == ".py" else "📓" if file_type == ".ipynb" else "📝"
        lines.append(f"**{i}.** {icon} `{fname}` — score: `{score}`\n> {snippet}…\n")
    return "\n".join(lines)


def handle_reindex(files) -> str:
    """Réindexe les documents"""
    if not files:
        return "⚠️ Aucun fichier sélectionné."
    
    docs_dir = Path("data/docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for uploaded_file in files:
        try:
            # Sauvegarder le fichier
            file_path = docs_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            success_count += 1
        except Exception as e:
            logger.error("Erreur copie %s : %s", uploaded_file.name, e)
    
    if success_count == 0:
        return "❌ Aucun fichier n'a pu être copié."
    
    try:
        with st.spinner("Réindexation en cours..."):
            run_ingestion(force=True)
            # Recharger le moteur
            st.cache_resource.clear()
            init_engine()
        return f"✅ {success_count} fichier(s) indexé(s) avec succès."
    except Exception as e:
        logger.error("Réindexation échouée : %s", e)
        return f"❌ Erreur : {e}"


# ── Chargement du moteur ──────────────────────────────────────────────────────
try:
    engine = init_engine()
except Exception as e:
    st.error(f"Erreur de chargement du moteur : {e}")
    st.stop()

# ── Interface principale ──────────────────────────────────────────────────────
# Header
st.markdown(f"""
<div class="pytutor-header">
    <h1>🐍 PyTutor</h1>
    <p>Assistant pédagogique Python — réponses appuyées sur votre base de documents.</p>
    <div class="model-badge">⚡ {LLM_MODEL}</div>
</div>
""", unsafe_allow_html=True)

# Colonnes : chat à gauche, sidebar à droite
chat_col, sidebar_col = st.columns([3, 1])

with chat_col:
    # Initialiser l'historique des messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Afficher l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources", expanded=False):
                    st.markdown(message["sources"])
    
    # Zone de saisie
    if prompt := st.chat_input("Posez votre question sur Python..."):
        # Ajouter le message utilisateur immédiatement
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Afficher un indicateur de chargement
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ Analyse en cours...")
            
            # Phase 1 : Classification
            if not is_python_question(prompt):
                logger.info("Question hors-sujet bloquée : %r", prompt[:80])
                bot_reply = OUT_OF_SCOPE_REPLY
                sources_md = "_Aucune source récupérée._"
            else:
                # Phase 2 : RAG
                try:
                    response = engine.chat(prompt)
                    bot_reply = str(response)
                    sources_md = format_sources(
                        response.source_nodes if hasattr(response, "source_nodes") else []
                    )
                except Exception as e:
                    logger.error("Erreur lors de la génération : %s", e)
                    bot_reply = f"❌ Une erreur s'est produite : {e}"
                    sources_md = "_Erreur_"
            
            # Mettre à jour le message
            message_placeholder.markdown(bot_reply)
            
            # Afficher les sources si disponibles
            if sources_md and sources_md != "_Aucune source récupérée._" and sources_md != "_Erreur_":
                with st.expander("📚 Sources", expanded=False):
                    st.markdown(sources_md)
        
        # Ajouter la réponse à l'historique
        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_reply,
            "sources": sources_md if sources_md and sources_md not in ["_Aucune source récupérée._", "_Erreur_"] else ""
        })
        
        # Rerun pour mettre à jour l'affichage
        st.rerun()

with sidebar_col:
    # Sidebar avec les sources de la dernière réponse
    st.markdown("### 📚 Sources")
    if st.session_state.messages and "sources" in st.session_state.messages[-1] and st.session_state.messages[-1]["sources"]:
        st.markdown(st.session_state.messages[-1]["sources"])
    else:
        st.markdown("_Les sources apparaîtront ici après chaque réponse._")
    
    st.divider()
    
    # Section documents
    st.markdown("### 📁 Documents")
    with st.expander("Ajouter / Réindexer", expanded=False):
        uploaded_files = st.file_uploader(
            "Fichiers : .pdf .py .md .txt .ipynb",
            type=["pdf", "py", "md", "txt", "ipynb"],
            accept_multiple_files=True
        )
        if st.button("🔄 Lancer l'indexation", type="primary"):
            if uploaded_files:
                result = handle_reindex(uploaded_files)
                st.success(result) if "✅" in result else st.error(result)
            else:
                st.warning("Sélectionnez des fichiers d'abord.")
    
    st.divider()
    
    # Bouton effacer la conversation
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()