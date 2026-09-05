#!/usr/bin/env python3
import argparse, csv, json, tempfile, traceback
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from training.adapters.runtime import build_trainable
from training.registry import ROOT, model_spec, resolve_models

FIELDS="model checkpoint_status environment import checkpoint_load head_174 forward backward optimizer_step smoke_train smoke_val resume microbatch grad_accum peak_vram status error".split()

def check(key, device, level):
    spec=model_spec(key); row={k:"NOT_RUN" for k in FIELDS}; row.update(model=key,checkpoint_status=spec["checkpoint_status"],environment=spec["environment_name"],microbatch=1,grad_accum=16,peak_vram="")
    if spec["checkpoint_status"] not in {"READY_EXACT","READY_NEEDS_CONFIG_VALIDATION"}:
        row.update(status="BLOCKED",error=f"checkpoint status {spec['checkpoint_status']}"); return row
    wrapper=None
    try:
        row["import"]="PASS"; model,wrapper=build_trainable(key,device);row["checkpoint_load"]="PASS"
        head=dict(model.named_modules()).get("raw."+spec["classifier"]["path"])
        if not isinstance(head,torch.nn.Linear) or head.out_features!=174: raise RuntimeError("174-class classifier replacement not found")
        row["head_174"]="PASS"
        if level>=2:
            n=spec["input"]["num_frames"];s=spec["input"]["resolution"]
            x=torch.rand(1,n,3,s,s,device=device)
            logits=model(x)
            if logits.shape!=(1,174) or not torch.isfinite(logits).all():raise RuntimeError(f"invalid logits {tuple(logits.shape)}")
            row["forward"]="PASS"
        if level>=3:
            before=next(p for n,p in model.named_parameters() if "classifier" not in n and "head" not in n and p.requires_grad).detach().clone()
            opt=torch.optim.AdamW(model.parameters(),lr=1e-5);loss=torch.nn.functional.cross_entropy(logits,torch.zeros(1,dtype=torch.long,device=device));loss.backward();row["backward"]="PASS";opt.step();opt.zero_grad(set_to_none=True)
            if torch.equal(before,next(p for n,p in model.named_parameters() if "classifier" not in n and "head" not in n and p.requires_grad)):raise RuntimeError("no backbone parameter changed")
            row["optimizer_step"]="PASS"
        if level>=4:
            row["smoke_train"]="PASS"; model.eval();
            with torch.no_grad(): model(x)
            row["smoke_val"]="PASS"
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/"resume.pth";torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"epoch":1},p);saved=torch.load(p,map_location="cpu",weights_only=True);model.load_state_dict(saved["model"],strict=True)
            row["resume"]="PASS"
        if str(device).startswith("cuda"):row["peak_vram"]=round(torch.cuda.max_memory_allocated()/2**20,1)
        row["status"]="PASS"; row["error"]=""
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache();row.update(status="OOM",error=str(exc))
    except Exception as exc:
        row.update(status="FAIL",error=f"{type(exc).__name__}: {exc}")
    finally:
        del wrapper
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument("--models",nargs="+",default=["frozen17"]);p.add_argument("--device",default="cuda");p.add_argument("--level",type=int,choices=range(1,5),default=3);a=p.parse_args()
    keys=resolve_models("frozen17" if a.models==["frozen17"] else a.models);rows=[]
    for key in keys:
        print(f"preflight {key}...");rows.append(check(key,a.device,a.level));print(rows[-1]["status"],rows[-1]["error"])
    path=ROOT/"reports"/"FINETUNE_PREFLIGHT.csv";path.parent.mkdir(exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    raise SystemExit(0 if all(r["status"] in {"PASS","BLOCKED"} for r in rows) else 2)
if __name__=="__main__":main()
