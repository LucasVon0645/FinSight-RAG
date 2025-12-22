from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# LangChain wrapper for HF pipeline (newer package)
from langchain_huggingface import HuggingFacePipeline


def build_hf_llm(model_id: str, max_new_tokens: int = 400, temperature: float = 0.2):
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