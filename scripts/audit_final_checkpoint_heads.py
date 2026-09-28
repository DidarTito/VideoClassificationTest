"""Inspect original checkpoint classifier tensors without allocating any GPU model."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from models import REQUESTED_MODEL_SPECS
from scripts.generate_ssv2_full17_device import read, sha
from scripts.ssv2_accuracy_evidence import EVIDENCE

def main():
    records = {}
    for row in read(ROOT / "results/final_k400_ssv2_master.csv"):
        if row["AccuracyValid"] == "TRUE":
            records[row["Checkpoint"]] = int(row["ClassifierClasses"])
    for key in EVIDENCE:
        records[REQUESTED_MODEL_SPECS[key].for_dataset("ssv2").checkpoint] = 174
    audit = []
    endings = ("head.weight", "head.proj.weight", "cls_head.fc_cls.weight", "classifier.weight", "video.1.weight")
    for relative, classes in records.items():
        path = ROOT / relative
        item = {"Checkpoint": relative, "SHA256": sha(path), "ExpectedClasses": classes}
        try:
            if path.suffix == ".safetensors":
                from safetensors import safe_open
                with safe_open(path, framework="pt", device="cpu") as payload:
                    heads = {key: payload.get_slice(key).get_shape() for key in payload.keys() if key.endswith(endings)}
            else:
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                except RuntimeError:
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                for container in ("state_dict", "model_state", "model", "module", "heads"):
                    if isinstance(payload, dict) and isinstance(payload.get(container), dict):
                        payload = payload[container]
                heads = {key: list(value.shape) for key, value in payload.items() if isinstance(value, torch.Tensor) and key.endswith(endings)}
                del payload
            item.update(HeadShapes=heads, Valid=bool(heads) and any(shape[0] == classes for shape in heads.values()))
        except Exception as exc:
            item.update(Valid=False, Error=f"{type(exc).__name__}: {exc}")
        audit.append(item)
        print(relative, item["Valid"], item.get("HeadShapes", item.get("Error")), flush=True)
    (ROOT / "results/final_checkpoint_head_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
