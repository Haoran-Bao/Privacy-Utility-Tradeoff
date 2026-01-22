from typing import Dict

from .convnet import SmallConvNet


def create_model(cfg: Dict):
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "convnet")
    num_classes = int(model_cfg.get("num_classes", 10))
    if name == "convnet":
        return SmallConvNet(num_classes=num_classes)
    raise ValueError(f"Unknown model name: {name}")
