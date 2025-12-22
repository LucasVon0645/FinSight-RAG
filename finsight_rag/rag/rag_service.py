from dataclasses import dataclass
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from finsight_rag.llms.text_generator import build_hf_llm as build_text_gen_llm

@dataclass
class RAGConfig:
    chroma_dir: str = "./annual_reports_chroma"
    collection: str = "brazilian_annual_reports"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    k: int = 5

    gen_model: str = "Qwen/Qwen2.5-3B-Instruct"
    temperature: float = 0.2
    max_new_tokens: int = 400


def format_docs(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source", meta.get("file_path", "unknown"))
        page = meta.get("page", meta.get("page_number", ""))
        blocks.append(f"[{i}] {src} page={page}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)

class RAGService:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

        # Must match your ingestion setup (same embedding + same persisted collection/dir)
        self.embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
        self.vector_store = Chroma(
            collection_name=cfg.collection,
            persist_directory=cfg.chroma_dir,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": cfg.k})

        # Local Transformers model wrapped for LangChain
        self.llm = build_text_gen_llm(
            model_id=cfg.gen_model,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )

        # Prompt (simple + reliable for RAG)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You answer using ONLY the provided context. "
             "If the answer is not in the context, say you don't know."),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer (cite like [1], [2]):")
        ])

        # LCEL pipeline:
        # - Take the user question
        # - Retrieve docs and format as context
        # - Fill prompt
        # - Generate answer with LLM
        # - Parse to string
        self.chain = (
            {
                "question": RunnablePassthrough(),
                "context": self.retriever | RunnableLambda(format_docs),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def answer(self, question: str, chat_history: Optional[List[Tuple[str, str]]] = None):
        # For a first version, ignore chat_history. (You can add it later.)
        text = self.chain.invoke(question)

        # If you want to show sources in Gradio, retrieve separately:
        docs = self.retriever.get_relevant_documents(question)
        return text, docs