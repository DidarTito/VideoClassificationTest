#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.datasets.ssv2 import audit_dataset
from training.registry import ROOT


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="datasets/ssv2/videos");p.add_argument("--labels",default="datasets/ssv2/labels/labels.json");p.add_argument("--train",default="third_party/UniFormer/video_classification/data_list/sthv2/somesomev2_rgb_train_split.txt");p.add_argument("--val",default="third_party/UniFormer/video_classification/data_list/sthv2/somesomev2_rgb_validation_split.txt");p.add_argument("--check-decode",action="store_true");a=p.parse_args()
    result=audit_dataset(a.root,a.labels,a.train,a.val,a.check_decode)
    ok=not result["overlap"] and not result["missing"]["train"] and not result["missing"]["val"] and not result["corrupt"]["train"] and not result["corrupt"]["val"]
    status="COMPLETE" if ok else ("TRAIN_AND_VAL_INCOMPLETE" if result["missing"]["train"] and result["missing"]["val"] else "TRAIN_INCOMPLETE" if result["missing"]["train"] else "VAL_INCOMPLETE")
    lines=["# SSV2 dataset audit","",f"- Status: **{status}**",f"- Classes: {result['classes']}",f"- Expected/present train videos: {result['train_count']}/{result['train_count']-len(result['missing']['train'])}",f"- Expected/present validation videos: {result['val_count']}/{result['val_count']-len(result['missing']['val'])}",f"- Missing train/validation videos: {len(result['missing']['train'])}/{len(result['missing']['val'])}",f"- Train/validation overlap: {len(result['overlap'])}",f"- Corrupt train/validation videos checked: {len(result['corrupt']['train'])}/{len(result['corrupt']['val'])}",f"- Annotation/cache identity SHA256: `{result['annotation_hash']}`",f"- Ready for training: **{ok}**","","Class IDs come only from the official JSON mapping; directory names and cache directories are never treated as classes."]
    reports=ROOT/"reports";reports.mkdir(exist_ok=True);(reports/"SSV2_DATASET_AUDIT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines[2:9])); raise SystemExit(0 if ok else 2)
if __name__=="__main__":main()
