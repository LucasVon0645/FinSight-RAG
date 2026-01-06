from importlib import resources as impresources

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# LangChain wrappers
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI

from finsight_rag.utils import load_yaml, get_gemini_api_key, get_hf_api_key
import finsight_rag.config as config

def build_hf_local_text_gen_llm(
    model_id: str,
    max_new_tokens: int = 400,
    temperature: float = 1.0,
):
    """
    Returns a LangChain ChatModel compatible with LCEL pipes.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
    )

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        return_full_text=False,
    )

    return HuggingFacePipeline(pipeline=gen_pipe)

def build_gemini_chat_llm(
    model_id: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_new_tokens: int = 400,
):
    """
    Returns a LangChain ChatModel compatible with LCEL pipes.
    """
    gemini_api_key = get_gemini_api_key()
    return ChatGoogleGenerativeAI(
        model=model_id,
        google_api_key=gemini_api_key,
        temperature=temperature,
        max_output_tokens=max_new_tokens,
    )

def build_hf_remote_chat_llm(
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    temperature: float = 0.0,
    max_new_tokens: int = 400,
):
    """
    Returns a LangChain LLM compatible with LCEL pipes
    using Hugging Face Inference API.
    """
    hf_token = get_hf_api_key()
    if not hf_token:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is not set")

    llm = HuggingFaceEndpoint(
        repo_id=model_id,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        huggingfacehub_api_token=hf_token,
        task="text-generation",
        provider="auto",
    )
    
    return ChatHuggingFace(llm=llm)

def get_chat_llm_from_cfg():
    """
    Create chat LLM from configuration file.
    """
    cfg_path = (impresources.files(config) / "chat_llm_config.yaml")
    # Load configuration from a YAML file or environment variables
    cfg = load_yaml(cfg_path)  # Example: adjust the path as needed
    model_id = cfg.get("model_id", "Qwen/Qwen2.5-7B-Instruct")
    temperature = cfg.get("temperature", 0.0)
    max_new_tokens = cfg.get("max_new_tokens", 400)

    if "gemini" in model_id.lower():
        return build_gemini_chat_llm(
            model_id=model_id,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    
    return build_hf_remote_chat_llm(
        model_id=model_id,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )