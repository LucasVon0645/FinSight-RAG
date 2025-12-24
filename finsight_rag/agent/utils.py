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

def extract_sources_from_docs(docs: List[Document]) -> List[Dict]:
    """
    Extract minimal, citation-friendly source metadata from a list of Documents.
    """
    sources = []

    for d in docs:
        md = d.metadata or {}

        sources.append({
            "source": (
                md.get("source")
                or md.get("path")
                or md.get("file_path")
            ),
            "page": md.get("page"),
            "company": md.get("company"),
            "year": md.get("year"),
        })

    return sources

def build_evidence_text_from_notes(notes: List[Dict]) -> str:
    """Build formatted evidence blocks from accumulated notes."""
    evidence_blocks = []
    for i, n in enumerate(notes, start=1):
        src_lines = []
        for s in n.get("sources", []):
            src_lines.append(
                f"- {s.get('company')} {s.get('year')} p.{s.get('page')}: {s.get('source')}"
            )
        evidence_blocks.append(
            f"[N{i}] Subquestion: {n.get('subquestion')}\n"
            f"Notes:\n{n.get('text')}\n"
            f"Sources:\n" + ("\n".join(src_lines) if src_lines else "- (none)")
        )
        
        evidence_text = "\n\n".join(evidence_blocks)
        
        return evidence_text