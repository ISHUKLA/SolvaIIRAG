import json
import os
from functools import lru_cache
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).with_name("RAG3_SolvencyII_improved.ipynb")

KEEP_MARKERS = [
    "from __future__ import annotations",
    "DOCS_DIR = Path(",
    "@dataclass",
    "def infer_metadata(",
    "SECTION_RE = re.compile(",
    "import fitz  # PyMuPDF",
    "def load_pdf(",
    "def load_docx(",
    "def load_text_like(",
    "def load_documents(",
    "def save_chunks(",
    "FRENCH_STOPWORDS = {",
    "class SolvencyRetriever:",
    "class SolvencyBM25Retriever:",
    "def format_citation(",
    "SYSTEM_PROMPT = \"\"\"",
    "def ask(",
    "def ask_bm25(",
    "EVAL_QUESTIONS = [",
    "def evaluate_retrieval(",
]


def _make_paths_portable(env: dict) -> None:
    project_dir = NOTEBOOK_PATH.parent
    index_dir = project_dir / "rag3_index"
    env["DOCS_DIR"] = project_dir / "Directive"
    env["INDEX_DIR"] = index_dir
    env["CHUNKS_PATH"] = index_dir / "chunks.jsonl"
    env["MANIFEST_PATH"] = index_dir / "manifest.json"
    env["AUDIT_LOG_PATH"] = project_dir / "audit_log.jsonl"


def _install_fast_hybrid_guards(env: dict) -> None:
    """Keep Hybrid queries responsive when the vector index is incomplete."""

    def _chunk_count() -> int:
        chunks_path = env.get("CHUNKS_PATH")
        if chunks_path is None or not chunks_path.exists():
            return 0
        return sum(1 for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip())

    def _vector_count() -> int:
        chromadb = env.get("chromadb")
        index_dir = env.get("INDEX_DIR")
        if chromadb is None or index_dir is None:
            return 0
        chroma_dir = index_dir / "chroma"
        if not chroma_dir.exists():
            return 0
        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))
            return client.get_collection("solvency_ii_chunks").count()
        except Exception:
            return 0

    def vector_index_ready() -> bool:
        expected = _chunk_count()
        return expected > 0 and _vector_count() == expected

    original_ask = env.get("ask")
    ask_bm25 = env.get("ask_bm25")
    if not callable(original_ask) or not callable(ask_bm25):
        return

    def ask_fast_hybrid(
        question: str,
        use_llm: bool = True,
        use_reranker: bool = False,
        audience: str = "expert",
        history: list[dict] | None = None,
    ) -> dict:
        if not vector_index_ready():
            result = ask_bm25(
                question,
                use_llm=use_llm,
                audience=audience,
                history=history,
            )
            result["mode_used"] = "BM25 fallback"
            result["hybrid_status"] = f"{_vector_count()}/{_chunk_count()}"
            return result

        return original_ask(
            question,
            use_llm=use_llm,
            use_reranker=use_reranker,
            audience=audience,
            history=history,
        )

    env["ask"] = ask_fast_hybrid
    env["vector_index_ready"] = vector_index_ready


@lru_cache(maxsize=1)
def load_runtime() -> dict:
    os.chdir(NOTEBOOK_PATH.parent)
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    env: dict[str, object] = {"__name__": "__main__"}

    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        if any(line.lstrip().startswith("!") for line in source.splitlines()):
            continue
        if not any(marker in source for marker in KEEP_MARKERS):
            continue
        exec(compile(source, f"{NOTEBOOK_PATH.name}#cell-{idx}", "exec"), env)

    _make_paths_portable(env)
    _install_fast_hybrid_guards(env)
    return env


def get_callable(name: str):
    runtime = load_runtime()
    return runtime[name]
