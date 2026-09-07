from time import perf_counter
from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.messages import BaseMessage
from langchain_huggingface import ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from finsight_rag.logging_config import logger, query_preview


def history_for_retrieval(messages: List[BaseMessage] | None) -> str:
    """Flatten structured chat messages into text for vector retrieval."""
    if not messages:
        return ""
    return "\n".join(message.content for message in messages)

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
        logger.info("RAGService initialized llm_type={}", type(llm).__name__)

        # Prompt (simple + reliable for RAG)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You answer using ONLY the provided context. "
             "If the answer is not in the context, say you don't know."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}")
        ])

        # LCEL pipeline:
        # - Take the user question
        # - Retrieve docs and format as context
        # - Fill prompt
        # - Generate answer with LLM
        # - Parse to string
        self.chain = (
            RunnablePassthrough()
            | {
                "question": lambda x: x["question"],
                "chat_history": lambda x: x.get("chat_history", []),
            }
            | RunnablePassthrough.assign(
                documents=lambda x: self.retriever.invoke(
                    f"{history_for_retrieval(x.get('chat_history'))}\n"
                    f"Current question: {x['question']}"
                )
            )
            | RunnablePassthrough.assign(
                answer=(
                    (lambda x: {
                        "question": x["question"],
                        "chat_history": x.get("chat_history", []),
                        "context": format_sources(x["documents"]),
                    })
                    | self.prompt
                    | self.llm
                    | StrOutputParser()
                )
            )
        )
        
        self.chain_from_context = (
            {
                "context": lambda x: x["context"],
                "question": lambda x: x["question"],
                "chat_history": lambda x: x.get("chat_history", []),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def retrieve(self, question: str, company: str | None = None) -> List[Document]:
        """Retrieve documents for the question. If company is given, filter to that company only."""
        started = perf_counter()
        logger.info(
            "RAGService.retrieve started question={} company_filter={}",
            query_preview(question),
            company or "none",
        )
        try:
            docs = self.retriever.invoke(question)
            retrieved_count = len(docs)

            if company:
                company = company.lower().strip()
                docs = [
                    d for d in docs
                    if (d.metadata or {}).get("company", "").lower().strip() == company
                ]

            logger.info(
                "RAGService.retrieve completed retrieved={} returned={} duration_ms={:.1f}",
                retrieved_count,
                len(docs),
                (perf_counter() - started) * 1000,
            )
            return docs
        except Exception:
            logger.exception("RAGService.retrieve failed question={}", query_preview(question))
            raise
    
    def answer(
        self,
        question: str,
        chat_history: List[BaseMessage] | None = None,
        return_docs: bool = False,
    ):
        """
        Generate an answer to the question using retrieved documents. The documents are
        retrieved internally from the retriever.
        Returns: (answer, sources_str) if return_docs is False, else (answer, docs)
        """

        started = perf_counter()
        logger.info(
            "RAGService.answer started question={} return_docs={}",
            query_preview(question),
            return_docs,
        )
        try:
            out = self.chain.invoke({"question": question, "chat_history": chat_history or []})
            answer: str = out["answer"]
            docs: List[Document] = out["documents"]

            logger.info(
                "RAGService.answer completed documents={} duration_ms={:.1f}",
                len(docs),
                (perf_counter() - started) * 1000,
            )

            if return_docs:
                return answer, docs

            sources_str = format_sources(docs)
            return answer, sources_str
        except Exception:
            logger.exception("RAGService.answer failed question={}", query_preview(question))
            raise
    
    def answer_from_docs(
        self,
        question: str,
        docs: List[Document],
        chat_history: List[BaseMessage] | None = None,
    ):
        """
        Generate an answer using ONLY the supplied docs (no retrieval).
        Returns: (answer, sources_str)
        """
        started = perf_counter()
        logger.info(
            "RAGService.answer_from_docs started question={} documents={}",
            query_preview(question),
            len(docs),
        )
        try:
            sources_str = format_sources(docs)
            out = self.chain_from_context.invoke(
                {
                    "question": question,
                    "context": sources_str,
                    "chat_history": chat_history or [],
                }
            )

            answer = out["answer"] if isinstance(out, dict) and "answer" in out else out
            logger.info(
                "RAGService.answer_from_docs completed duration_ms={:.1f}",
                (perf_counter() - started) * 1000,
            )
            return answer, sources_str
        except Exception:
            logger.exception(
                "RAGService.answer_from_docs failed question={}",
                query_preview(question),
            )
            raise