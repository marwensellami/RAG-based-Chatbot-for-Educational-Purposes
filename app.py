"""
app.py — Interface Gradio — PyTutor
Streaming : message utilisateur affiché immédiatement + réponse progressive
"""
from __future__ import annotations

import logging
import shutil
import os
from pathlib import Path

import gradio as gr
import requests

from src.chat_engine import build_chat_engine
from src.config import LLM_MODEL, OLLAMA_BASE_URL
from src.ingestion import run_ingestion, run_ingestion_if_needed

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

run_ingestion_if_needed()
engine = build_chat_engine()


# ══════════════════════════════════════════════════════════════════════════════
# Classificateur Python
# ══════════════════════════════════════════════════════════════════════════════

_CLASSIFIER_MODEL   = os.getenv("CLASSIFIER_MODEL", "qwen2.5:0.5b")
_CLASSIFIER_RETRIES = 3
_CLASSIFIER_TIMEOUT = 60

_CLASSIFIER_SYSTEM = """\
Tu es un classificateur binaire strict.
Réponds UNIQUEMENT par OUI si la question porte sur Python (langage, bibliothèques, outils),
ou NON dans tous les autres cas. Aucune explication. Juste OUI ou NON.
"""


def _call_classifier(message: str, model: str) -> str | None:
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


def _is_python_question(message: str) -> bool:
    for model in list(dict.fromkeys([_CLASSIFIER_MODEL, LLM_MODEL])):
        for attempt in range(1, _CLASSIFIER_RETRIES + 1):
            answer = _call_classifier(message, model)
            if answer is not None:
                is_python = answer.startswith("OUI")
                logger.info("Classificateur [%s] → %s", model, "Python" if is_python else "Hors-sujet")
                return is_python
    logger.error("Classificateur indisponible — fail-open.")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _format_sources(nodes) -> str:
    if not nodes:
        return "_Aucune source récupérée._"
    lines = ["### 📚 Sources utilisées\n"]
    for i, node in enumerate(nodes, 1):
        meta      = node.metadata or {}
        fname     = meta.get("file_name", "source inconnue")
        score     = f"{node.score:.3f}" if node.score else "—"
        snippet   = node.get_content()[:250].replace("\n", " ").strip()
        file_type = meta.get("file_type", "")
        icon      = "📄" if file_type == ".pdf" else "💻" if file_type == ".py" else "📓" if file_type == ".ipynb" else "📝"
        lines.append(f"**[{i}] {icon} `{fname}`** — score: `{score}`\n> {snippet}…\n")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Submit avec affichage immédiat + indicateur "En cours…"
# ══════════════════════════════════════════════════════════════════════════════

def _submit(msg: str, hist: list):
    """
    Générateur en 3 phases :
      Phase 1 — Affiche le message utilisateur immédiatement
      Phase 2 — Affiche "⏳ Analyse en cours…" pendant le traitement
      Phase 3 — Remplace par la vraie réponse
    """
    if not msg or not msg.strip():
        yield hist, hist, "", ""
        return

    # ── Phase 1 : message utilisateur affiché immédiatement ──────────────────
    pending = hist + [
        {"role": "user",      "content": msg},
        {"role": "assistant", "content": "⏳ Analyse en cours…"},
    ]
    yield pending, hist, "", ""

    # ── Phase 2 : classification + RAG ───────────────────────────────────────
    if not _is_python_question(msg):
        logger.info("Question hors-sujet bloquée : %r", msg[:80])
        bot_reply  = OUT_OF_SCOPE_REPLY
        sources_md = "_Aucune source récupérée._"
    else:
        response   = engine.chat(msg)
        bot_reply  = str(response)
        sources_md = _format_sources(
            response.source_nodes if hasattr(response, "source_nodes") else []
        )

    # ── Phase 3 : remplacer "En cours…" par la vraie réponse ─────────────────
    new_hist = hist + [
        {"role": "user",      "content": msg},
        {"role": "assistant", "content": bot_reply},
    ]
    yield new_hist, new_hist, "", sources_md


def clear_chat():
    engine.reset()
    logger.info("Historique effacé.")
    return [], []


def handle_reindex(files) -> str:
    global engine
    if not files:
        return "⚠️ Aucun fichier sélectionné."
    docs_dir = Path("data/docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0
    for f in files:
        path = f if isinstance(f, str) else f.name
        try:
            shutil.copy2(path, docs_dir / os.path.basename(path))
            success_count += 1
        except Exception as e:
            logger.error("Erreur copie %s : %s", path, e)
    if success_count == 0:
        return "❌ Aucun fichier n'a pu être copié."
    try:
        run_ingestion(force=True)
        engine = build_chat_engine()
        return f"✅ {success_count} fichier(s) indexé(s) avec succès."
    except Exception as e:
        logger.error("Réindexation échouée : %s", e)
        return f"❌ Erreur : {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Interface Gradio
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """
footer { display: none !important; }

#header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2040 100%);
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 20px;
    border: 1px solid #2d5a9e;
}
#header h1 { margin: 0 0 6px 0; font-size: 1.8rem; color: #e8f4ff; }
#header p  { margin: 0; color: #94b8d8; font-size: 0.92rem; }
#model-badge {
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

#chatbot { height: 520px; border-radius: 12px !important; border: 1px solid #2d3748 !important; }
#input-row { align-items: flex-end; gap: 8px; }
#msg-input textarea { border-radius: 10px !important; font-size: 0.95rem; }
#send-btn  { height: 76px !important; border-radius: 10px !important; font-size: 1rem !important; font-weight: 600 !important; }
#clear-btn { width: 100%; border-radius: 8px !important; margin-top: 4px !important; }
#sidebar   { padding-left: 10px; }
.sidebar-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: #718096; font-weight: 600; margin: 12px 0 4px 0; }
.source-box { font-size: 0.83rem; line-height: 1.6; }
"""

with gr.Blocks(title="PyTutor - Assistant Python", css=_CSS) as demo:

    gr.HTML(f"""
    <div id="header">
        <h1>🐍 PyTutor</h1>
        <p>Assistant pédagogique Python — réponses appuyées sur votre base de documents.</p>
        <div id="model-badge">⚡ {LLM_MODEL}</div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Conversation",
                height=520,
                avatar_images=(
                    None,
                    "https://api.dicebear.com/7.x/bottts/svg?seed=pytutor"
                ),
            )
            with gr.Row(elem_id="input-row"):
                msg_input = gr.Textbox(
                    placeholder="Ex : Explique-moi les décorateurs en Python…",
                    label="", show_label=False,
                    lines=3, max_lines=6, scale=5,
                    elem_id="msg-input", autofocus=True,
                )
                send_btn = gr.Button(
                    "Envoyer ➤", variant="primary",
                    scale=1, elem_id="send-btn",
                )
            clear_btn = gr.Button(
                "🗑️ Effacer la conversation",
                variant="secondary", elem_id="clear-btn",
            )

        with gr.Column(scale=1, elem_id="sidebar"):
            gr.HTML("<div class='sidebar-title'>📚 Sources</div>")
            with gr.Accordion("Voir les sources", open=True):
                sources_md = gr.Markdown(
                    value="_Les sources apparaîtront ici après chaque réponse._",
                    elem_classes=["source-box"],
                )
            gr.HTML("<div class='sidebar-title'>📁 Documents</div>")
            with gr.Accordion("Ajouter / Réindexer", open=False):
                file_upload = gr.File(
                    label="Fichiers : .pdf .py .md .txt .ipynb",
                    file_types=[".pdf", ".py", ".md", ".txt", ".ipynb"],
                    file_count="multiple",
                )
                reindex_btn    = gr.Button("🔄 Lancer l'indexation", variant="secondary")
                reindex_status = gr.Textbox(
                    label="Statut", interactive=False,
                    lines=2, placeholder="En attente…",
                )

    history_state = gr.State([])

    send_btn.click(
        _submit,
        inputs=[msg_input, history_state],
        outputs=[chatbot, history_state, msg_input, sources_md],
    )
    msg_input.submit(
        _submit,
        inputs=[msg_input, history_state],
        outputs=[chatbot, history_state, msg_input, sources_md],
    )
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, history_state],
    )
    reindex_btn.click(
        handle_reindex,
        inputs=[file_upload],
        outputs=[reindex_status],
    )


if __name__ == "__main__":
    demo.launch(share=True)
