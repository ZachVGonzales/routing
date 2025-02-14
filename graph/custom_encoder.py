from semantic_router.encoders import BaseEncoder
from pydantic.v1 import PrivateAttr
from typing import Any, List, Optional


class STEncoder(BaseEncoder):
  name : str = "multi-qa-mpnet-base-dot-v1"
  score_threshold : float = 0.5
  type : str = "sentence-transformer"
  device : Optional[str] = None
  _model : Any = PrivateAttr()
  _torch : Any = PrivateAttr()

  def __init__(self, name : str = None, score_threshold : float = None, **data):
    super().__init__(**data)
    
    if name:
      self.name = name
    
    if score_threshold:
      self.score_threshold = score_threshold
      
    self._model = self._initialize_st_model()

  def _initialize_st_model(self):
    try:
      from sentence_transformers import SentenceTransformer
    except ImportError:
      raise ImportError(
          "Please install sentence_transformers to use STEncoder. "
          "You can install it with: "
          "`pip install sentence_transformers`"
        )
    
    try:
      import torch
    except ImportError:
      raise ImportError(
        "Please install Pytorch to use STEncoder. "
        "You can install it with: "
        "`pip install torch`"
      )
    
    if self.device is None:
      self._torch = torch
      self.device = "cuda" if self._torch.cuda.is_available() else "cpu"

    return SentenceTransformer(self.name, device=self.device)
  
  def __call__(self, docs: List[str], batch_size: int = 32, normalize_embeddings: bool = True) -> List[List[float]]:
    embeddings = self._model.encode(docs, batch_size=batch_size, normalize_embeddings=normalize_embeddings)

    if embeddings.shape[0] == 0:  # Ensure embeddings exist
      raise ValueError("STEncoder produced empty embeddings. Check input data.")

    embeddings = embeddings.tolist()
    return embeddings