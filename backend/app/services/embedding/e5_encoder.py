from typing import Any


class E5TextEncoder:
    """
    Encoder cho model E5.

    Document phai co prefix `passage: `, query phai co prefix `query: `.
    Class nay chi lo encode text thanh vector normalized float32.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        max_length: int = 512,
        device: str | None = None,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing semantic dependencies. Install torch and transformers "
                "to use E5 embedding."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype).to(
            self.device
        )
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 16) -> Any:
        try:
            import numpy as np
            import torch.nn.functional as F
        except ImportError as exc:
            raise RuntimeError("Missing numpy/torch dependencies for embedding.") from exc

        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start: start + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with self.torch.no_grad():
                with self.torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                    outputs = self.model(**inputs)

                embeddings = self._mean_pooling(
                    outputs.last_hidden_state,
                    inputs["attention_mask"],
                )
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy().astype("float32"))

            del inputs, outputs, embeddings
            if self.device == "cuda":
                self.torch.cuda.empty_cache()

        return np.vstack(all_embeddings)

    def _mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = self.torch.sum(last_hidden_state.float() * mask, dim=1)
        counts = self.torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts
