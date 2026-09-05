#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch, yaml
from torch.utils.data import DataLoader
from training.adapters.runtime import build_trainable
from training.amp import autocast, amp_policy
from training.datasets.ssv2 import SSV2Dataset
from training.registry import ROOT,FROZEN17_KEYS,model_spec
from training.validation import validate

def main():
 p=argparse.ArgumentParser();p.add_argument("--model",required=True,choices=FROZEN17_KEYS);p.add_argument("--checkpoint",required=True);p.add_argument("--split",default="val",choices=["val"]);p.add_argument("--device",default="cuda");a=p.parse_args();spec=model_spec(a.model);base=yaml.safe_load((ROOT/"configs/ssv2/default.yaml").read_text());model,owner=build_trainable(a.model,a.device);state=torch.load(a.checkpoint,map_location="cpu",weights_only=True);model.load_state_dict(state["model"],strict=True);ds=SSV2Dataset(ROOT/base["videos"],ROOT/base["val_annotations"],ROOT/base["labels"],spec["input"]["num_frames"],spec["input"]["resolution"]);loader=DataLoader(ds,batch_size=1,num_workers=base["num_workers"],pin_memory=True);dtype,_=amp_policy(a.device,spec["finetune"]["amp_mode"]);metrics=validate(model,loader,a.device,lambda:autocast(a.device,dtype));h=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest();result={**metrics,"failed_or_corrupt":0,"checkpoint_sha256":h,"model":a.model,"split":"val","input":spec["input"]};print(json.dumps(result,indent=2));Path(a.checkpoint).with_name("val_summary.json").write_text(json.dumps(result,indent=2))
if __name__=="__main__":main()
