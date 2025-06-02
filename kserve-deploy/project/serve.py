import json
import base64
import logging
import numpy as np
import torch
import torch.nn.functional as F
import kserve
import re
from vllm import LLM

from typing import List, Tuple, Dict, Optional
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from sentence_transformers import CrossEncoder, SentenceTransformer
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(msecs)d %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class E5Model(kserve.Model):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        name = re.sub(r"[/-]", "_", model_name).lower()
        super().__init__(name)
        self.name = name
        self.model_name = model_name
        self.ready = False
        self.model = None
        self.tokenizer = None
        self.config = None
        self.gpu = torch.cuda.is_available()
        self.device = "cuda" if self.gpu else "cpu"
        self.load()

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model = self.model.eval().to(self.device)
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.ready = True

    @staticmethod
    def average_pool(
        last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def predict(
        self, request_data: Dict, request_headers: Optional[Dict] = None
    ) -> Dict:
        texts: List[Tuple[str, str]] = request_data["texts"]
        texts = [f"{text_type}: {text}" for text_type, text in texts]

        batch_dict = self.tokenizer(
            texts,
            max_length=self.config.max_position_embeddings,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**batch_dict)
        embeddings = self.average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return {"embeddings": embeddings.cpu().tolist()}


class BgeM3(kserve.Model):
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        name = re.sub(r"[/-]", "_", model_name).lower()
        super().__init__(name)
        self.name = name
        self.model_name = model_name
        self.ready = False
        self.llm = None
        self.load()

    def load(self) -> None:
        self.llm = LLM(model=self.model_name, task="embed")
        self.ready = True

    def predict(
        self, request_data: Dict, request_headers: Optional[Dict] = None
    ) -> Dict:
        texts: List[Tuple[str, str]] = request_data["texts"]
        texts_for_model = [text for _, text in texts]
        outputs = [self.llm.embed(text) for text in texts_for_model]
        embeddings = [output.outputs.embedding for output in outputs]
        return {"embeddings": embeddings}


class E5InstructModel(kserve.Model):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct"):
        name = re.sub(r"[/-]", "_", model_name).lower()
        super().__init__(name)
        self.name = name
        self.model_name = model_name
        self.ready = False
        self.model = None
        self.tokenizer = None
        self.config = None
        self.gpu = torch.cuda.is_available()
        self.device = "cuda" if self.gpu else "cpu"
        self.load()

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model = self.model.eval().to(self.device)
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.ready = True

    @staticmethod
    def average_pool(
        last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @staticmethod
    def get_detailed_instruct(query: str) -> str:
        return f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}"

    def predict(
        self, request_data: Dict, request_headers: Optional[Dict] = None
    ) -> Dict:
        texts: List[Tuple[str, str]] = request_data["texts"]
        texts = [
            self.get_detailed_instruct(text) if text_type.lower() == "query" else text
            for text_type, text in texts
        ]

        batch_dict = self.tokenizer(
            texts,
            max_length=self.config.max_position_embeddings,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**batch_dict)

        embeddings = self.average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return {"embeddings": embeddings.cpu().tolist()}


class CrossEncoderModelAmberoad(kserve.Model):
    def __init__(
        self, model_name: str = "amberoad/bert-multilingual-passage-reranking-msmarco"
    ):
        name = re.sub(r"[/-]", "_", model_name).lower()
        super().__init__(name)
        self.model_name = model_name
        self.name = name
        self.ready = False
        self.model = None
        self.gpu = torch.cuda.is_available()
        self.device = "cuda" if self.gpu else "cpu"
        self.load()

    def load(self) -> None:
        self.model = CrossEncoder(self.model_name, device=self.device)
        self.ready = True

    def predict(self, request_data: Dict, request_headers: Dict) -> Dict:
        instances = request_data["texts"]
        with torch.inference_mode():
            scores = self.model.predict(instances, convert_to_numpy=True)
        if len(scores.shape) > 1:
            scores = scores[:, 1]
        return {"scores": scores.tolist()}


class CrossEncoderRerankerV2M3(kserve.Model):
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        name = re.sub(r"[/-]", "_", model_name).lower()
        super().__init__(name)
        self.model_name = model_name
        self.name = name
        self.ready = False
        self.llm = None
        self.load()

    def load(self) -> None:
        self.llm = LLM(model=self.model_name, task="score")
        self.ready = True

    def predict(
        self, request_data: Dict, request_headers: Optional[Dict] = None
    ) -> Dict:
        pairs = request_data["texts"]
        outputs = [self.llm.score(pair[0], pair[1]) for pair in pairs]
        scores = [output.outputs.score for output in outputs]
        return {"scores": scores}


if __name__ == "__main__":
    #  e5model = E5Model()
    #  e5model_instruct = E5InstructModel()
    # e5model_reranker = CrossEncoderModel()
    embedder = BgeM3()
    reranker = CrossEncoderRerankerV2M3()
    kserve.ModelServer(http_port=8100, enable_docs_url=True).start(
        models=[embedder, reranker]
    )
