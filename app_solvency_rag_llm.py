from __future__ import annotations

import html
import logging
import os
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config - must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Solvabilité II · RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --bg: #0d1117;
        --surface: #161b22;
        --surface-soft: #1c232d;
        --border: #30363d;
        --accent: #c9a84c;
        --accent2: #5b8dd9;
        --text: #e6edf3;
        --muted: #8b949e;
        --danger: #f85149;
        --ok: #3fb950;
        --radius: 12px;
        --radius-sm: 7px;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif;
    }

    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    .app-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 26px 0 18px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 24px;
    }
    .app-header .icon { font-size: 2.4rem; line-height: 1; }
    .app-header h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 1.95rem;
        font-weight: 400;
        color: var(--text);
        margin: 0;
        letter-spacing: 0;
    }
    .badge {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        background: var(--accent);
        color: #000;
        padding: 2px 8px;
        border-radius: 999px;
        margin-left: 5px;
        vertical-align: middle;
    }
    .sub {
        font-size: 0.82rem;
        color: var(--muted);
        margin: 3px 0 0;
    }

    .legal-disclaimer {
        position: sticky;
        top: 0;
        z-index: 30;
        background: #241f12;
        border: 1px solid rgba(201,168,76,.65);
        border-left: 4px solid var(--accent);
        border-radius: var(--radius-sm);
        color: var(--text);
        padding: 11px 14px;
        margin: 0 0 22px;
        font-size: 0.84rem;
        line-height: 1.45;
        box-shadow: 0 10px 24px rgba(0,0,0,.22);
    }
    .legal-disclaimer strong {
        color: var(--accent);
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 999px;
        border: 1px solid;
    }
    .status-pill.ok { color: var(--ok); border-color: var(--ok); background: rgba(63,185,80,.08); }
    .status-pill.err { color: var(--danger); border-color: var(--danger); background: rgba(248,81,73,.08); }
    .status-pill.warn { color: var(--accent); border-color: var(--accent); background: rgba(201,168,76,.08); }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

    .sidebar-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.3px;
        text-transform: uppercase;
        color: var(--muted);
        margin: 18px 0 7px;
    }

    .metric-box {
        background: var(--surface-soft);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 13px 12px;
        text-align: center;
    }
    .metric-box .val {
        font-family: 'DM Serif Display', serif;
        font-size: 1.7rem;
        color: var(--accent);
    }
    .metric-box .lbl {
        font-size: 0.72rem;
        color: var(--muted);
        margin-top: 1px;
    }

    .msg-user {
        background: rgba(91,141,217,.12);
        border: 1px solid rgba(91,141,217,.25);
        border-radius: var(--radius);
        padding: 14px 18px;
        margin: 12px 0 8px;
        font-size: 0.95rem;
    }
    .msg-assistant {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 14px 18px;
        margin-bottom: 12px;
        font-size: 0.95rem;
        line-height: 1.65;
    }
    .msg-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 6px;
    }

    .source-card {
        background: rgba(255,255,255,.03);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: var(--radius-sm);
        padding: 12px 14px;
        margin-bottom: 8px;
        font-size: 0.82rem;
        color: var(--muted);
        line-height: 1.5;
    }
    .source-title {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        color: var(--accent);
        margin-bottom: 5px;
        font-weight: 500;
    }
    .citation-strip {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid var(--border);
    }
    .citation-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 7px;
    }
    .citation-pill {
        display: inline-block;
        margin: 0 6px 6px 0;
        padding: 4px 8px;
        border: 1px solid rgba(201,168,76,.45);
        border-radius: 999px;
        background: rgba(201,168,76,.08);
        color: var(--accent);
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        line-height: 1.35;
    }

    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(201,168,76,.15) !important;
    }

    [data-testid="stButton"] > button,
    [data-testid="stDownloadButton"] > button {
        background: var(--accent) !important;
        color: #000 !important;
        font-weight: 700 !important;
        font-family: 'DM Sans', sans-serif !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 8px 18px !important;
        transition: opacity .15s, transform .1s !important;
    }
    [data-testid="stButton"] > button:hover,
    [data-testid="stDownloadButton"] > button:hover {
        opacity: .88 !important;
        transform: translateY(-1px) !important;
    }
    [data-testid="stButton"] > button:disabled {
        opacity: .45 !important;
        transform: none !important;
    }

    [data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }

    hr { border-color: var(--border) !important; margin: 20px 0 !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Safe runtime import
# ---------------------------------------------------------------------------
try:
    from solvency_notebook_runtime import load_runtime  # type: ignore

    _RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    _RUNTIME_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_QUESTION_LENGTH = 1_000
DOCS_DIR = PROJECT_DIR / "Directive"

SUGGESTED_QUESTIONS = [
    "Best Estimate - définition ?",
    "Risk Margin et calcul ?",
    "Art. 45 - ORSA ?",
    "Exigences de gouvernance ?",
    "SCR - Article 101 ?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_html(value: object) -> str:
    return html.escape(str(value)).replace("\n", "<br>")


def history_to_csv(history: list[dict]) -> str:
    if not history:
        return ""
    rows = [
        {
            "question": item["question"],
            "answer": item["answer"],
            "n_sources": len(item.get("chunks", [])),
            "scope_status": item.get("scope_status", "in_scope"),
            "scope_reason": item.get("scope_reason", ""),
            "matched_terms": ", ".join(item.get("matched_terms", [])),
        }
        for item in history
    ]
    buf = StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue()


def _chunk_score_label(chunk: object, retrieval_scores: dict | None) -> str:
    if not retrieval_scores:
        return ""
    score = retrieval_scores.get(getattr(chunk, "chunk_id", ""))
    if score is None:
        return ""
    return f" · pertinence {float(score):.2f}"


def render_chunks(chunks: list, retrieval_scores: dict | None = None, max_chars: int = 1_600) -> None:
    for i, chunk in enumerate(chunks, 1):
        page = f" · p.{chunk.page}" if getattr(chunk, "page", None) else ""
        section = f" · {chunk.section}" if getattr(chunk, "section", None) else ""
        score = _chunk_score_label(chunk, retrieval_scores)
        text = getattr(chunk, "text", "")
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        title = f"[{i}] {getattr(chunk, 'source_name', 'Source')}{page}{section}{score}"
        st.markdown(
            f"""<div class="source-card">
                <div class="source-title">{_safe_html(title)}</div>
                {_safe_html(text)}
            </div>""",
            unsafe_allow_html=True,
        )


def render_inline_citations(chunks: list, retrieval_scores: dict | None = None, limit: int = 6) -> str:
    citations: list[str] = []
    seen: set[tuple[str, object]] = set()

    for chunk in chunks:
        source = getattr(chunk, "source_name", "Source")
        page = getattr(chunk, "page", None)
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        page_label = f"p. {page}" if page else "page n/a"
        score_label = _chunk_score_label(chunk, retrieval_scores)
        citations.append(f"{source} · {page_label}{score_label}")

    if not citations:
        return ""

    pills = "".join(
        f'<span class="citation-pill">{_safe_html(citation)}</span>'
        for citation in citations[:limit]
    )
    extra = len(citations) - limit
    if extra > 0:
        pills += f'<span class="citation-pill">+{extra} autre(s)</span>'

    return (
        '<div class="citation-strip">'
        '<div class="citation-label">Citations</div>'
        f"{pills}"
        "</div>"
    )


def _pill(label: str, kind: str = "ok") -> str:
    return f'<span class="status-pill {kind}"><span class="dot"></span>{label}</span>'


def render_scope_badge(item: dict) -> str:
    status = item.get("scope_status", "in_scope")
    reason = item.get("scope_reason", "")
    matched_terms = item.get("matched_terms", [])
    if status == "out_of_scope":
        label = "Hors contexte"
        kind = "err"
    elif status == "uncertain":
        label = "Périmètre incertain"
        kind = "warn"
    else:
        label = "Périmètre Solvabilité II"
        kind = "ok"

    details = reason
    if matched_terms:
        details += " · " + ", ".join(matched_terms[:6])
    if details:
        label = f"{label} · {details}"
    return _pill(_safe_html(label), kind)


def _secret_value(name: str, default: str = "") -> str:
    """Read a deployed Streamlit secret first, then fall back to environment variables."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = os.environ.get(name, default)
    return str(value or default)


def validate_groq_connection(groq_key: str, model: str = "llama-3.3-70b-versatile") -> tuple[bool, str]:
    """Return a visible health check for optional Groq generation."""
    key = groq_key.strip()
    if not key:
        return False, "Clé GROQ_API_KEY manquante."
    try:
        from langchain_groq import ChatGroq
    except Exception as exc:
        return False, f"Package langchain_groq indisponible: {type(exc).__name__}: {exc}"

    previous_key = os.environ.get("GROQ_API_KEY")
    os.environ["GROQ_API_KEY"] = key
    try:
        llm = ChatGroq(model=model, temperature=0, max_tokens=8, timeout=20)
        response = llm.invoke("Réponds seulement: OK")
        content = str(getattr(response, "content", "")).strip()
        return True, content or "OK"
    except Exception as exc:
        return False, f"Appel Groq impossible avec {model}: {type(exc).__name__}: {exc}"
    finally:
        if previous_key is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = previous_key


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_runtime() -> dict:
    if not _RUNTIME_AVAILABLE:
        raise RuntimeError(
            "Module `solvency_notebook_runtime` introuvable. "
            "Cette app cherche le module dans le dossier du projet."
        )
    return load_runtime()


def run_query(
    question: str,
    mode: str,
    use_reranker: bool,
    audience: str,
    history: list[dict],
    groq_key: str,
) -> dict:
    use_llm = bool(groq_key.strip())
    if use_llm:
        os.environ["GROQ_API_KEY"] = groq_key.strip()
    else:
        os.environ.pop("GROQ_API_KEY", None)

    runtime = get_runtime()
    if mode == "BM25":
        return runtime["ask_bm25"](question, use_llm=use_llm, audience=audience, history=history)
    return runtime["ask"](
        question,
        use_llm=use_llm,
        use_reranker=use_reranker,
        audience=audience,
        history=history,
    )


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "history": [],
        "runtime_ready": False,
        "runtime_error": None,
        "question_input": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()

if not _RUNTIME_AVAILABLE:
    st.error(
        "**Module manquant** : `solvency_notebook_runtime` n'est pas trouvé. "
        "Cette app cherche le module dans le dossier du projet."
    )
    st.stop()

if not st.session_state.runtime_ready:
    with st.spinner("⏳ Chargement de l'index Solvabilité II..."):
        try:
            get_runtime()
            st.session_state.runtime_ready = True
            st.session_state.runtime_error = None
            logger.info("Runtime loaded automatically.")
        except Exception as exc:
            st.session_state.runtime_ready = False
            st.session_state.runtime_error = f"{type(exc).__name__}: {exc}"
            st.error(f"Erreur : {exc}")
            logger.exception("Automatic runtime init failed.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<p style="font-family:\'DM Serif Display\',serif;font-size:1.25rem;'
        'margin:0 0 4px;color:#e6edf3;">⚖️ Solvabilité II</p>'
        '<p style="font-size:0.75rem;color:#8b949e;margin:0 0 20px;">RAG · Assistant réglementaire public</p>',
        unsafe_allow_html=True,
    )

    groq_key = _secret_value("GROQ_API_KEY")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key.strip()

    st.markdown('<p class="sidebar-label">Corpus embarqué</p>', unsafe_allow_html=True)
    docs_dir = DOCS_DIR
    if docs_dir.exists():
        n_docs = len([path for path in docs_dir.glob("*") if path.is_file()])
        st.caption(f"Directive/ · {n_docs} fichier(s) détecté(s)")
    else:
        st.error("Corpus Directive/ introuvable dans le repository.")

    st.markdown('<p class="sidebar-label">Paramètres</p>', unsafe_allow_html=True)
    search_mode_labels = {
        "Hybride": "Recherche intelligente · recommandée",
        "BM25": "Mots exacts · références précises",
    }
    mode = st.selectbox(
        "Mode de recherche",
        ["Hybride", "BM25"],
        index=0,
        format_func=lambda value: search_mode_labels[value],
        help=(
            "La recherche intelligente combine les mots exacts et le sens de la question. "
            "Les mots exacts sont utiles pour retrouver un article, un code, un acronyme "
            "ou une expression précise."
        ),
    )
    if mode == "Hybride":
        st.caption("Recommandé : recherche les mots exacts et les idées proches.")
    else:
        st.caption("Utile pour un article, un code, un acronyme ou une phrase exacte.")
    use_reranker = st.toggle(
        "Utiliser le reranker",
        value=False,
        disabled=(mode != "Hybride"),
        help="Disponible en recherche intelligente, mais plus lent.",
    )
    show_sources = st.toggle("Afficher les sources", value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.runtime_ready:
        st.markdown(_pill("Moteur actif", "ok"), unsafe_allow_html=True)
    elif st.session_state.runtime_error:
        st.markdown(_pill("Erreur moteur", "err"), unsafe_allow_html=True)
        st.caption(st.session_state.runtime_error)
    else:
        st.markdown(_pill("En attente", "warn"), unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.history:
        n = len(st.session_state.history)
        avg_src = sum(len(h.get("chunks", [])) for h in st.session_state.history) // max(n, 1)
        col1, col2 = st.columns(2)
        col1.markdown(
            f'<div class="metric-box"><div class="val">{n}</div><div class="lbl">Questions</div></div>',
            unsafe_allow_html=True,
        )
        col2.markdown(
            f'<div class="metric-box"><div class="val">{avg_src}</div><div class="lbl">Sources</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

    csv = history_to_csv(st.session_state.history)
    st.download_button(
        "⬇ Exporter l'historique",
        data=csv or " ",
        file_name="historique_solvency_rag.csv",
        mime="text/csv",
        disabled=not bool(csv),
        use_container_width=True,
    )

    if st.button("🗑 Effacer la conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="icon">⚖️</div>
        <div>
            <h1>Solvabilité II <span class="badge">RAG</span></h1>
            <p class="sub">Interrogez la directive avec citations. La génération LLM via Groq est optionnelle.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="legal-disclaimer">
        <strong>Avertissement légal.</strong>
        SolvA2RAG est un outil d'aide à la recherche documentaire et à la synthèse.
        Il ne remplace ni la lecture des textes officiels, ni la vérification des versions applicables,
        ni l'analyse d'un professionnel qualifié.
    </div>
    """,
    unsafe_allow_html=True,
)

if not groq_key.strip():
    st.info("Mode sans API : l'app répond avec les passages les plus pertinents et leurs citations.")
else:
    st.info(
        "Mode LLM actif avec Groq. Premier chargement : l'index peut prendre 1 à 2 minutes. "
        "Ensuite, les questions suivantes sont beaucoup plus rapides."
    )

st.markdown("**Suggestions rapides**")
chip_cols = st.columns(len(SUGGESTED_QUESTIONS))
for col, q in zip(chip_cols, SUGGESTED_QUESTIONS):
    with col:
        if st.button(q, key=f"chip_{q}", use_container_width=True):
            st.session_state.question_input = q
            st.rerun()

st.markdown("---")
audience_label = st.radio(
    "Style de réponse",
    [
        "🎓 Expert (citations formelles)",
        "💬 Vulgarisé (analogies + exemples)",
    ],
    horizontal=True,
)
audience = "vulgarise" if audience_label.startswith("💬") else "expert"

st.text_area(
    "Votre question",
    label_visibility="collapsed",
    placeholder="Ex : Que dit l'article 101 sur le SCR ?",
    max_chars=MAX_QUESTION_LENGTH,
    height=95,
    key="question_input",
)

question = st.session_state.question_input.strip()
col_send, col_info = st.columns([1, 5])

with col_send:
    send = st.button("Envoyer ↵", use_container_width=True, disabled=not bool(question))

with col_info:
    st.caption(f"{len(st.session_state.question_input)}/{MAX_QUESTION_LENGTH} caractères")

if send:
    if not docs_dir.exists():
        st.error("Corpus Directive/ introuvable dans le repository. Vérifie le déploiement.")
    elif st.session_state.runtime_error:
        st.error(f"Index réglementaire indisponible : {st.session_state.runtime_error}")
    else:
        spinner_text = "Recherche des sources et génération LLM..." if groq_key.strip() else "Recherche des sources..."
        with st.spinner(spinner_text):
            try:
                result = run_query(
                    question=question,
                    mode=mode,
                    use_reranker=use_reranker,
                    audience=audience,
                    history=st.session_state.history,
                    groq_key=groq_key,
                )
                st.session_state.history.append(result)
                st.session_state.runtime_ready = True
                st.session_state.runtime_error = None
                logger.info("Query answered: %s", question[:80])
                st.rerun()
            except Exception as exc:
                st.session_state.runtime_error = f"{type(exc).__name__}: {exc}"
                st.error(f"Erreur lors de la génération : {type(exc).__name__}: {exc}")
                logger.exception("Query failed: %s", question[:80])

if st.session_state.history:
    st.markdown("---")
    st.markdown("**Dernière réponse**")
    for index, item in enumerate(reversed(st.session_state.history)):
        if index == 1:
            st.markdown("**Historique précédent**")

        st.markdown(
            f'<div class="msg-user"><div class="msg-label">Vous</div>{_safe_html(item["question"])}</div>',
            unsafe_allow_html=True,
        )
        retrieval_scores = item.get("retrieval_scores", {})
        citations_html = render_inline_citations(item.get("chunks", []), retrieval_scores)
        scope_html = render_scope_badge(item)
        st.markdown(
            f'<div class="msg-assistant"><div class="msg-label">Assistant</div>'
            f'{scope_html}<br><br>{citations_html}{_safe_html(item["answer"])}</div>',
            unsafe_allow_html=True,
        )
        feedback_key = len(st.session_state.history) - index
        feedback_state_key = f"feedback_message_{feedback_key}"
        col_useful, col_not_useful, col_feedback = st.columns([0.35, 0.35, 5.3])
        with col_useful:
            if st.button("👍", key=f"useful_{feedback_key}", help="Utile"):
                st.session_state[feedback_state_key] = "Merci de votre retour"
        with col_not_useful:
            if st.button("👎", key=f"not_useful_{feedback_key}", help="Pas utile"):
                st.session_state[feedback_state_key] = "Nous allons améliorer"
        with col_feedback:
            if st.session_state.get(feedback_state_key):
                st.caption(st.session_state[feedback_state_key])

        if show_sources and item.get("chunks"):
            with st.expander(f"📄 Voir les extraits complets ({len(item['chunks'])} source(s))", expanded=False):
                render_chunks(item["chunks"], retrieval_scores)
