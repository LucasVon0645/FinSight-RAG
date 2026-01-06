import os
from importlib import resources as impresources
from typing import List


from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import finsight_rag.config as config
from finsight_rag.utils import load_yaml
from finsight_rag.ingest.utils import extract_company_from_filename, extract_year_from_filename

class VectorStoreWrapper:
    """Wrapper for Chroma vector store."""

    def __init__(self):
        rag_config_path = (impresources.files(config) / "rag_config.yaml")
        rag_config_dict = load_yaml(rag_config_path)
        
        chroma_dir = rag_config_dict["vector_store_path"]
        embedding_model_id = rag_config_dict["embedding_model"]
        top_k_chunks = rag_config_dict.get("k", 5)
        
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_id)
        self.vector_store = Chroma(
            collection_name="brazilian_annual_reports",
            persist_directory=chroma_dir,
            embedding_function=self.embedding_model,
        )
        
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k_chunks})
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        self.vector_store.add_documents(documents)

    def query(self, query_text: str, top_k: int = 5) -> List[Document]:
        """Query the vector store and return top_k documents."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        return retriever.get_relevant_documents(query_text)
    
    def get_retriever(self):
        """Get the retriever for the vector store."""
        return self.retriever
    
    def add_document_from_filepath(self, file_path: str):
        """Add a single document from a file path."""
        
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        filename = os.path.basename(file_path)
        
        # Extract metadata
        company = extract_company_from_filename(filename)
        year = extract_year_from_filename(filename)
        
        for d in docs:
            d.metadata["company"] = company
            d.metadata["file_name"] = filename
            if year is not None:
                d.metadata["year"] = year

        self.add_documents(docs)
        
        status = (
            f"📄 Loaded **{len(docs)}** pages\n\n"
            f"🏢 company: `{company}`\n\n"
            f"📅 year: `{year if year is not None else 'not found'}`"
        )
        
        return status
        