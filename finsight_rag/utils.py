import yaml

def load_yaml(file_path: str) -> dict:
    """Load a YAML file and return its contents as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_data_config(config_path: str) -> dict:
    """Load data configuration from a YAML file."""
    return load_yaml(config_path)

def get_gemini_api_key() -> str:
    """Retrieve the Gemini API key from file."""
    file_path = "C:\\Users\\User\\projects\\FinSight-RAG\\gemini_api_key"
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def get_hf_api_key() -> str:
    """Retrieve the Hugging Face API key from file."""
    file_path = "C:\\Users\\User\\projects\\FinSight-RAG\\hf_api_key"
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()