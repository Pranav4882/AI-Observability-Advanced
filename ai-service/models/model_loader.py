import numpy as np
import onnxruntime as ort
import json
import os
from typing import Tuple, Dict, Optional, List

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class MockNNModule:
        pass
    class nn:
        Module = MockNNModule

MODEL_CLASS_NAMES: Dict[str, List[str]] = {
    "fashion_mnist": [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ],
    "mnist_digits": [str(d) for d in range(10)],
    "sentiment": ["negative", "positive"],
}

class SingleModel:
    def __init__(self, name: str, class_names: List[str]):
        self.name = name
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.onnx_session: Optional[ort.InferenceSession] = None
        self.pytorch_model: Optional[nn.Module] = None
        self.training_baseline: Optional[Dict] = None
        self.model_version: str = "unknown"
        self.runtime_type: str = "none"

    def load_onnx(self, path: str) -> bool:
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            self.onnx_session = ort.InferenceSession(
                path, opts, providers=["CPUExecutionProvider"]
            )
            self.runtime_type = "onnx"
            print(f"  [{self.name}] onnx ok ({path})")
            return True
        except Exception as e:
            print(f"  [{self.name}] onnx load failed: {e}")
            return False

    def load_pytorch(self, path: str, model_cls) -> bool:
        if not HAS_TORCH:
            print(f"  [{self.name}] skipping pytorch (not installed)")
            return False
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.pytorch_model = model_cls()
            if "model_state_dict" in ckpt:
                self.pytorch_model.load_state_dict(ckpt["model_state_dict"])
                self.model_version = ckpt.get("model_version", "unknown")
            else:
                self.pytorch_model.load_state_dict(ckpt)
            self.pytorch_model.eval()
            self.runtime_type = "pytorch"
            print(f"  [{self.name}] pytorch ok ({path})")
            return True
        except Exception as e:
            print(f"  [{self.name}] pytorch load error: {e}")
            return False

    def load_baseline(self, path: str) -> bool:
        try:
            with open(path, "r") as f:
                self.training_baseline = json.load(f)
            print(f"  [{self.name}] baseline loaded")
            return True
        except Exception as e:
            print(f"  [{self.name}] no baseline: {e}")
            return False

    def predict(self, input_tensor: np.ndarray) -> Tuple[int, str, float, list]:
        if self.onnx_session is not None:
            return self._predict_onnx(input_tensor)
        elif self.pytorch_model is not None:
            return self._predict_pytorch(input_tensor)
        raise RuntimeError(f"no model for '{self.name}'")

    def _predict_onnx(self, t: np.ndarray):
        inp = self.onnx_session.get_inputs()[0].name
        logits = self.onnx_session.run(None, {inp: t})[0][0]
        return self._process_logits(logits)

    def _predict_pytorch(self, t: np.ndarray):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not installed but _predict_pytorch was called!")
        with torch.no_grad():
            logits = self.pytorch_model(torch.from_numpy(t))[0].numpy()
        return self._process_logits(logits)

    def _process_logits(self, logits: np.ndarray):
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        cls = int(np.argmax(probs))
        conf = float(probs[cls])
        name = self.class_names[cls]
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [
            {"class": int(i), "name": self.class_names[i],
             "probability": float(probs[i])}
            for i in top3_idx
        ]
        return cls, name, conf, top3

class ModelManager:
    def __init__(self):
        self.models: Dict[str, SingleModel] = {}
        self.default_model: str = "fashion_mnist"

    def load_all(self, models_dir: str = "./models"):
        print("loading models...")
        model_configs = [
            ("fashion_mnist", "v2.0.0"),
            ("mnist_digits", "v1.0.0"),
            ("sentiment", "v1.0.0"),
        ]
        for name, version in model_configs:
            class_names = MODEL_CLASS_NAMES.get(name, [])
            if not class_names:
                continue

            sm = SingleModel(name, class_names)
            sm.model_version = version

            onnx_path    = os.path.join(models_dir, f"{name}.onnx")
            pth_path     = os.path.join(models_dir, f"{name}.pth")
            baseline_path = os.path.join(models_dir, f"{name}_baseline.json")

            if os.path.exists(onnx_path):
                sm.load_onnx(onnx_path)
            if sm.runtime_type == "none" and os.path.exists(pth_path):
                if not HAS_TORCH:
                    print(f"  ⚠️ [{name}] PyTorch skipped (torch not installed)")
                    continue
                from models.train import FashionCNN, DigitCNN, SentimentMLP
                arch_map = {
                    "fashion_mnist": FashionCNN,
                    "mnist_digits": DigitCNN,
                    "sentiment": SentimentMLP,
                }
                cls = arch_map.get(name)
                if cls:
                    sm.load_pytorch(pth_path, cls)

            if os.path.exists(baseline_path):
                sm.load_baseline(baseline_path)

            if sm.runtime_type != "none":
                self.models[name] = sm

        loaded = list(self.models.keys())
        print(f"ready: {loaded}")
        if loaded and self.default_model not in loaded:
            self.default_model = loaded[0]

    def get(self, name: str) -> Optional[SingleModel]:
        return self.models.get(name)

    def predict(self, input_tensor: np.ndarray,
                model_name: Optional[str] = None) -> Tuple[int, str, float, list]:
        name = model_name or self.default_model
        m = self.models.get(name)
        if m is None:
            raise RuntimeError(
                f"unknown model '{name}' (have: {list(self.models.keys())})"
            )
        return m.predict(input_tensor)

    def list_models(self) -> List[Dict]:
        return [
            {
                "name": m.name,
                "version": m.model_version,
                "runtime": m.runtime_type,
                "num_classes": m.num_classes,
                "classes": m.class_names,
            }
            for m in self.models.values()
        ]

    @property
    def runtime_type(self):
        m = self.models.get(self.default_model)
        return m.runtime_type if m else "none"

    @property
    def model_version(self):
        m = self.models.get(self.default_model)
        return m.model_version if m else "unknown"

    @property
    def training_baseline(self):
        m = self.models.get(self.default_model)
        return m.training_baseline if m else None

model_manager = ModelManager()
