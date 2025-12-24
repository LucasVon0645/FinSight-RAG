from dataclasses import dataclass
from typing import List
from importlib import resources as impresources

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import finsight_rag.config as config
from finsight_rag.utils import load_yaml

from finsight_rag.llms.llm_service import (
    build_hf_remote_chat_llm,
    build_gemini_chat_llm,
)


@dataclass
class RAGConfig:
    chroma_dir: str = "./annual_reports_chroma"
    collection: str = "brazilian_annual_reports"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    k: int = 5

    gen_model: str = "Qwen/Qwen2.5-3B-Instruct"
    temperature: float = 1.0
    max_new_tokens: int = 400


def format_sources(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source", meta.get("file_path", "unknown"))
        page = meta.get("page", meta.get("page_number", ""))
        year = meta.get("year", "")
        company = meta.get("company", "")
        blocks.append(f"[{i}] {src} page={page} year={year} company={company}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)

def join_docs_contents(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

def format_chat_history(chat_history):
    if not chat_history:
        return ""
    return "\n".join(
        f"Human: {q}\nAssistant: {a}"
        for q, a in chat_history
    )

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
        
        if "gemini" in cfg.gen_model.lower():
            build_chat_llm = build_gemini_chat_llm
        else:
            build_chat_llm = build_hf_remote_chat_llm

        # Local Transformers model wrapped for LangChain
        self.llm = build_chat_llm(
            model_id=cfg.gen_model,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )

        # Prompt (simple + reliable for RAG)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You answer using ONLY the provided context. "
             "If the answer is not in the context, say you don't know."),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}")
        ])

        # LCEL pipeline:
        # - Take the user question
        # - Retrieve docs and format as context
        # - Fill prompt
        # - Generate answer with LLM
        # - Parse to string
        self.chain = (
            RunnablePassthrough()  # incoming is the question string
            | {"question": RunnablePassthrough()}
            | RunnablePassthrough.assign(documents=lambda x: self.retriever.invoke(x["question"]))
            | RunnablePassthrough.assign(
                answer=(
                    (lambda x: {"question": x["question"], "context": format_sources(x["documents"])})
                    | self.prompt
                    | self.llm
                    | StrOutputParser()
                )
            )
        )
        
        self.chain_from_context = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def retrieve(self, question: str, company: str | None = None) -> List[Document]:
        """Retrieve documents for the question. If company is given, filter to that company only."""
        docs = self.retriever.invoke(question)
        
        if company:
            company = company.lower().strip()
            docs = [d for d in docs if (d.metadata or {}).get("company", "").lower().strip() == company]
        return docs
    
    def answer(self, question: str, return_docs: bool = False):
        # Generate answer
        out = self.chain.invoke(question)
        
        answer = out["answer"]
        docs = out["documents"]
        
        if return_docs:
            return answer, docs
        
        sources_str = format_sources(docs)
        return answer, sources_str
    
    def answer_from_docs(self, question: str, docs: list[Document]):
        """
        Generate an answer using ONLY the supplied docs (no retrieval).
        Returns: (answer, sources_str)
        """
        sources_str = format_sources(docs)

        # use the same prompt template you already use, but bypass retriever
        out = self.chain_from_context.invoke({"question": question, "context": sources_str})

        answer = out["answer"] if isinstance(out, dict) and "answer" in out else out
        return answer, sources_str

def get_rag_service() -> RAGService:
    # --- RAG setup ---
    rag_config_path = (impresources.files(config) / "rag_config.yaml")
    rag_config_dict = load_yaml(rag_config_path)

    chroma_dir = rag_config_dict["vector_store_path"]
    gen_model = rag_config_dict["gen_model"]
    embedding_model = rag_config_dict["embedding_model"]
    gen_model = rag_config_dict["gen_model"]
    temperature = rag_config_dict["temperature"]
    max_new_tokens = rag_config_dict["max_new_tokens"]
    top_k_chunks = rag_config_dict["top_k_chunks"]

    rag_service = RAGService(RAGConfig(
        chroma_dir=chroma_dir,
        collection="brazilian_annual_reports",
        embedding_model=embedding_model,
        gen_model=gen_model,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        k=top_k_chunks,
    ))
    
    return rag_service
