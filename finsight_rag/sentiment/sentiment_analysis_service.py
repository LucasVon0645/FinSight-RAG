from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


TextOrTexts = Union[str, List[str]]

@dataclass
class SentimentPrediction:
    label: str
    score: float


class SentimentAnalysisService:
    """
    Loads a fine-tuned HF sequence classification model from disk and provides
    a simple predict() API.

    - Uses config dict loaded from sentiment_analyser_config.yaml
    - Loads model from inference.model_dir if present, else training.output_dir
    - CPU-only safe
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        model_dir = (
            cfg.get("inference", {}).get("model_dir")
            or cfg.get("training", {}).get("output_dir")
        )
        if not model_dir:
            raise ValueError("Could not determine model_dir from config (inference.model_dir or training.output_dir).")

        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        # Model/tokenizer were saved in the same folder by Trainer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.eval()

        # CPU-only (but keep this explicit)
        self.device = torch.device("cpu")
        self.model.to(self.device)

        self.max_length = int(cfg.get("model", {}).get("max_length", 256))

        inf = cfg.get("inference", {})
        self.batch_size = int(inf.get("batch_size", 16))
        self.return_all_scores = bool(inf.get("return_all_scores", True))
        self.top_k = inf.get("top_k", None)  # None => return all scores; 1 => top label only

        # Use model config label mapping if available
        # (Trainer saves config.json; may include id2label/label2id)
        self.id2label = inf.get("id2label", {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
        self.num_labels = int(self.model.config.num_labels)

    def _softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """ Computes softmax over the last dimension of logits tensor. """
        # Numerically stable softmax
        logits = logits - logits.max(dim=-1, keepdim=True).values
        exp = torch.exp(logits)
        return exp / exp.sum(dim=-1, keepdim=True)

    def predict(self, texts: TextOrTexts) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Returns predictions for one text or a list of texts.

        Output format (per text):
        {
          "label": "LABEL_2",
          "score": 0.87,
          "scores": [
            {"label": "LABEL_0", "score": 0.03},
            {"label": "LABEL_1", "score": 0.10},
            {"label": "LABEL_2", "score": 0.87}
          ]
        }
        """
        single_input = isinstance(texts, str)
        batch = [texts] if single_input else list(texts)

        outputs: List[Dict[str, Any]] = []

        with torch.no_grad():
            for i in range(0, len(batch), self.batch_size):
                chunk = batch[i : i + self.batch_size]

                enc = self.tokenizer(
                    chunk,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}

                logits = self.model(**enc).logits.detach().cpu()
                probs = self._softmax(logits)

                for p in probs:
                    top_idx = int(torch.argmax(p).item())
                    top_label = self.id2label.get(top_idx, f"LABEL_{top_idx}")
                    top_score = float(p[top_idx])

                    # Build full score list
                    score_list = [
                        {"label": self.id2label.get(j, f"LABEL_{j}"), "score": float(p[j])}
                        for j in range(self.num_labels)
                    ]
                    score_list.sort(key=lambda x: x["score"], reverse=True)

                    if self.top_k == 1 and not self.return_all_scores:
                        outputs.append({"label": top_label, "score": top_score})
                    else:
                        outputs.append({"label": top_label, "score": top_score, "scores": score_list})

        return outputs[0] if single_input else outputs
