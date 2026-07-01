import numpy as np
from PIL import Image
import io
import hashlib
import json
import time
import os
from typing import Tuple

_NORM = {
    "fashion_mnist": (0.2860, 0.3530),
    "mnist_digits":  (0.1307, 0.3081),
}
_VOCAB_SIZE = 512
_vocab_cache = None

def preprocess_image(image_bytes: bytes, model_name: str = "fashion_mnist") -> Tuple[np.ndarray, str, float]:
    start = time.perf_counter()
    mean, std = _NORM.get(model_name, (0.5, 0.5))
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(image, dtype=np.float32) / 255.0

    border_pixels = []
    border_pixels.extend(arr[0, :])
    border_pixels.extend(arr[-1, :])
    border_pixels.extend(arr[1:-1, 0])
    border_pixels.extend(arr[1:-1, -1])
    if np.mean(border_pixels) > 0.5:
        arr = 1.0 - arr
    arr = (arr - mean) / std
    tensor = arr.reshape(1, 1, 28, 28).astype(np.float32)
    input_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
    ms = (time.perf_counter() - start) * 1000
    return tensor, input_hash, ms

def _get_vocab() -> dict:
    global _vocab_cache
    if _vocab_cache is None:
        vocab_path = os.path.join(os.path.dirname(__file__), "..", "models", "sentiment_vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                _vocab_cache = json.load(f)
        else:
            _vocab_cache = {}
    return _vocab_cache

def preprocess_text(text: str) -> Tuple[np.ndarray, str, float]:
    start = time.perf_counter()
    vocab = _get_vocab()
    vec = np.zeros(_VOCAB_SIZE, dtype=np.float32)
    for w in text.lower().split():
        if w in vocab: vec[vocab[w]] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0: vec /= norm
    tensor = vec.reshape(1, _VOCAB_SIZE).astype(np.float32)
    input_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    ms = (time.perf_counter() - start) * 1000
    return tensor, input_hash, ms
