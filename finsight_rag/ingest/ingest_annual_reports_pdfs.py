import hashlib
from dataclasses import dataclass
import os
from typing import Iterable, List

from importlib import resources as impresources

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import finsight_rag.config as config
from finsight_rag.utils import load_yaml

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

@dataclass
class IngestConfig:
    corpus_path: str  # e.g.  ./data/annual_reports/pdf/

    # Chunking
    chunk_size: int = 400
    chunk_overlap: int = 50
    min_chars_per_page: int = 100
    
    # Batching
    batch_rows: int = 50  # rows -> explode to many docs; keep modest

    # Vector store (Chroma local)
    chroma_dir: str = "./annual_reports_chroma"
    collection: str = "brazilian_annual_reports"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

class AnnualReportsIngestor:
    """
    Docstring for AnnualReportsIngestor
    Ingestor for company annual reports in PDF format.
    """
    def __init__(self, cfg: IngestConfig):
        self.cfg = cfg
                
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        
        self.embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
        self.vector_store = Chroma(
            collection_name=cfg.collection,
            persist_directory=cfg.chroma_dir,
            embedding_function=self.embeddings,
        )
    
    def load_pdfs(self) -> Iterable[Document]:
        """
        Load PDFs from the corpus path and return as a list of Documents.
        """
        documents = []
        
        for root, _, files in os.walk(self.cfg.corpus_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = root + "/" + file
                    print(f"Loading PDF: {pdf_path}")
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    documents.extend(docs)
        
        min_chars_per_page = self.cfg.min_chars_per_page  # adjust as needed

        filtered_documents = []

        for doc in documents:
            text = doc.page_content.strip()
            if len(text) >= min_chars_per_page:
                filtered_documents.append(doc)
                
        return filtered_documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents into smaller pieces using the text splitter.
        """
        chunked_docs = []
        for doc in documents:
            chunks = self.splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                chunked_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={**doc.metadata, "chunk_index": i}
                    )
                )
        return chunked_docs
    
    def add_docs_to_vector_store(self, documents: List[Document]) -> None:
        """
        Add documents to the Chroma vector store.
        """
        self.vector_store.add_documents(documents)
        # self.vector_store.persist()
    
    def ingest(self):
        """
        Main ingestion method to load, chunk, and store documents.
        """
        
        print("Loading PDFs...")
        documents = self.load_pdfs()
        print(f"Loaded {len(documents)} documents from PDFs.")
        
        print("Chunking documents...")
        chunked_docs = self.chunk_documents(documents)
        print(f"Chunked into {len(chunked_docs)} documents.")
        
        print("Adding documents to vector store...")
        self.add_docs_to_vector_store(chunked_docs)
        print("Ingestion complete.")

if __name__ == "__main__":
    rag_config_path = (impresources.files(config) / "rag_config.yaml")

    rag_config = load_yaml(rag_config_path)
    dataset_path = rag_config["dataset_path"]
    vector_store_path = rag_config["vector_store_path"]
    embedding_model = rag_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    
    print("Ingesting dataset...")
    print(f"Dataset path: {dataset_path}")
    print(f"Vector store path: {vector_store_path}")
    print(f"Embedding model: {embedding_model}")

    cfg = IngestConfig(
        corpus_path=dataset_path,
        chroma_dir=vector_store_path,
        collection="brazilian_annual_reports",
        chunk_size=rag_config["chunk_size"],
        chunk_overlap=rag_config["chunk_overlap"],
        batch_rows=rag_config["batch_rows"],
        embedding_model=embedding_model,
    )
    AnnualReportsIngestor(cfg).ingest()
    print("Done.")