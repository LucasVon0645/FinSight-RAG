import os
from langchain_core.documents import Document
from typing import List, Dict

def dedupe_docs(docs: List[Document]) -> List[Document]:
    """Deduplicate documents based on source and page number metadata."""
    seen = set()
    out = []
    for d in docs:
        meta = d.metadata or {}
        key = (meta.get("source", meta.get("file_path", "")), meta.get("page", meta.get("page_number", "")))
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

def format_sources(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", meta.get("file_path", "unknown"))
        src = os.path.basename(str(src))  # filename only
        page = meta.get("page", meta.get("page_number", ""))
        year = meta.get("year", "")

        blocks.append(
            f"{src} page={page} year={year}\n{d.page_content}"
        )
    return "\n\n---\n\n".join(blocks)