from dataclasses import dataclass
from typing import List, Optional
from importlib import resources as impresources

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
    """
    RAG Service for document retrieval and question answering.
    Uses a vector store retriever (ex Chroma) and a chat LLM (HuggingFace or Gemini).
    """
    def __init__(self,
                 vector_store_retriever: VectorStoreRetriever,
                 llm: ChatHuggingFace | ChatGoogleGenerativeAI):

        
        
        self.retriever = vector_store_retriever

        self.llm = llm

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
        """
        Generate an answer to the question using retrieved documents. The documents are
        retrieved internally from the retriever.
        Returns: (answer, sources_str) if return_docs is False, else (answer, docs)
        """

        out = self.chain.invoke(question)
        
        answer: str = out["answer"]
        docs: List[Document] = out["documents"]
        
        if return_docs:
            return answer, docs
        
        sources_str = format_sources(docs)
        return answer, sources_str
    
    def answer_from_docs(self, question: str, docs: List[Document]):
        """
        Generate an answer using ONLY the supplied docs (no retrieval).
        Returns: (answer, sources_str)
        """
        sources_str = format_sources(docs)

        # use the same prompt template you already use, but bypass retriever
        out = self.chain_from_context.invoke({"question": question, "context": sources_str})

        answer = out["answer"] if isinstance(out, dict) and "answer" in out else out
        return answer, sources_str