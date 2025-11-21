import yaml
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def task(self) -> str:
        return self.raw["task"]

    @property
    def dataset(self) -> Dict[str, Any]:
        return self.raw["dataset"]

    @property
    def teacher_model_name(self) -> str:
        return self.raw["teacher_model_name"]

    @property
    def student_model_name(self) -> str:
        return self.raw["student_model_name"]

    @property
    def max_length(self) -> int:
        return int(self.raw.get("max_length", 128))

    @property
    def batch_size(self) -> int:
        return int(self.raw.get("batch_size", 32))

    @property
    def num_labels(self) -> int:
        return int(self.raw.get("num_labels", 2))

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def distillation(self) -> Dict[str, Any]:
        return self.raw.get("distillation", {})

    @property
    def output_dir(self) -> str:
        return self.raw["output_dir"]


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(data)
