import os
import yaml
from importlib import resources as impresources
import finsight_rag.config as config

def load_yaml(file_path: str) -> dict:
    """Load a YAML file and return its contents as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_data_config(config_path: str) -> dict:
    """Load data configuration from a YAML file."""
    return load_yaml(config_path)

def get_gemini_api_key() -> str:
    """Retrieve the Gemini API key from file."""
    file_path = str(impresources.files(config) / "gemini_api_key")
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def get_hf_api_key() -> str:
    """Retrieve the Hugging Face API key from file."""
    
    file_path = str(impresources.files(config) / "hf_api_key")
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def list_local_pdfs() -> list[str]:
    """List all PDF files in the local PDFs directory."""
    rag_config_path = str(impresources.files(config) / "rag_config.yaml")
    rag_config = load_yaml(str(rag_config_path))
    pdfs_dir_path = rag_config.get("dataset_path", "data/pdfs")
    if not os.path.isdir(pdfs_dir_path):
        return []
    files = [f for f in os.listdir(pdfs_dir_path) if f.lower().endswith(".pdf")]
    files.sort(key=lambda x: x.lower())
    return files

def get_local_pdfs_dir() -> str:
    """Get the local PDFs directory from RAG configuration."""
    rag_config_path = str(impresources.files(config) / "rag_config.yaml")
    rag_config = load_yaml(str(rag_config_path))
    local_pdfs_dir: str = rag_config.get("dataset_path", "data/pdfs")
    return local_pdfs_dir

def get_pdf_path(selected_filename: str):
    """Get the full path of a selected PDF file."""
    if not selected_filename:
        return None
    pdf_dir = get_local_pdfs_dir()
    path = os.path.join(pdf_dir, selected_filename)
    return path if os.path.exists(path) else None