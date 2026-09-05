#!/usr/bin/env python3
import argparse, json, os, platform, signal, subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch, yaml
from torch.utils.data import DataLoader, Subset

from training.adapters.runtime import build_trainable
from training.amp import amp_policy, autocast
from training.datasets.ssv2 import SSV2Dataset, audit_dataset
from training.engine import append_metrics, train_one_epoch
from training.registry import ROOT, FROZEN17_KEYS, model_spec
from training.scheduler import cosine_with_warmup
from training.validation import validate

def config_for(key):
    base=yaml.safe_load((ROOT/"configs/ssv2/default.yaml").read_text()); own=yaml.safe_load((ROOT/model_spec(key)["finetune"]["config"]).read_text());base.update({k:v for k,v in own.items() if k!="base"});return base

def main():
    p=argparse.ArgumentParser();p.add_argument("--model",required=True,choices=FROZEN17_KEYS);p.add_argument("--device",default="cuda");p.add_argument("--smoke",action="store_true");p.add_argument("--allow-partial-data",action="store_true");p.add_argument("--run-id");p.add_argument("--force",action="store_true");a=p.parse_args();spec=model_spec(a.model);cfg=config_for(a.model)
    audit=audit_dataset(ROOT/cfg["videos"],ROOT/cfg["labels"],ROOT/cfg["train_annotations"],ROOT/cfg["val_annotations"])
    missing=len(audit["missing"]["train"])+len(audit["missing"]["val"])
    partial = bool(audit["overlap"] or missing or audit["corrupt"]["train"] or audit["corrupt"]["val"])
    if partial and not (a.smoke and a.allow_partial_data): raise SystemExit(f"Dataset validation failed: overlap={len(audit['overlap'])}, missing={missing}. Full training requires complete SSV2; use --smoke --allow-partial-data only for a non-final diagnostic.")
    run_id=a.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run=ROOT/"runs/ssv2"/a.model/run_id
    if (run/"best.pth").exists() and not a.force: raise SystemExit(f"Completed run exists: {run}")
    run.mkdir(parents=True,exist_ok=True); (run/"resolved_config.yaml").write_text(yaml.safe_dump({**cfg,"model_spec":spec},sort_keys=False),encoding="utf-8")
    commit=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True).stdout.strip();(run/"git_commit.txt").write_text(commit+"\n")
    (run/"environment.txt").write_text(f"python={platform.python_version()}\ntorch={torch.__version__}\ncuda={torch.version.cuda}\ndevice={a.device}\n")
    (run/"checkpoint_source.json").write_text(json.dumps({k:spec.get(k) for k in ("k400_checkpoint","checkpoint_sha256","checkpoint_origin","checkpoint_status","source_repo","source_commit")},indent=2))
    (run/"protocol_status.json").write_text(json.dumps({"protocol_deviation": partial, "final_result": not partial, "status": "NONFINAL_PARTIAL_SSV2_SMOKE" if partial else "FINAL_ELIGIBLE"}, indent=2))
    model,owner=build_trainable(a.model,a.device)
    (run/"checkpoint_metadata.json").write_text(json.dumps({"load_policy":"backend strict load; only declared 400-way classifier replaced","classifier_path":spec["classifier"]["path"],"source_classes":400,"target_classes":174,"missing_backbone_keys":[],"unexpected_backbone_keys":[]},indent=2))
    dtype,scaler=amp_policy(a.device,spec["finetune"]["amp_mode"]);ctx=lambda:autocast(a.device,dtype)
    train=SSV2Dataset(ROOT/cfg["videos"],ROOT/cfg["train_annotations"],ROOT/cfg["labels"],spec["input"]["num_frames"],spec["input"]["resolution"]);val=SSV2Dataset(ROOT/cfg["videos"],ROOT/cfg["val_annotations"],ROOT/cfg["labels"],spec["input"]["num_frames"],spec["input"]["resolution"])
    if partial:
        train=Subset(train,[i for i, (video_id, _) in enumerate(train.samples) if (Path(cfg["videos"]) / f"{video_id}.webm").is_file()]);val=Subset(val,[i for i, (video_id, _) in enumerate(val.samples) if (Path(cfg["videos"]) / f"{video_id}.webm").is_file()])
        if not len(train) or not len(val): raise SystemExit("Partial SSV2 smoke requires at least one available train and validation video")
    if a.smoke: train=Subset(train,range(min(64,len(train))));val=Subset(val,range(min(32,len(val))));cfg["epochs"]=1
    workers=int(cfg["num_workers"]);kwargs=dict(batch_size=int(cfg["microbatch"]),num_workers=workers,pin_memory=str(a.device).startswith("cuda"),persistent_workers=workers>0 and bool(cfg["persistent_workers"]));
    if workers:kwargs["prefetch_factor"]=int(cfg["prefetch_factor"])
    train_loader=DataLoader(train,shuffle=True,**kwargs);val_loader=DataLoader(val,shuffle=False,**kwargs)
    opt=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]));scheduler=cosine_with_warmup(opt,int(cfg["epochs"]),int(cfg["warmup_epochs"]));start=0;best=-1.0
    last=run/"last.pth"
    if last.exists():
        state=torch.load(last,map_location="cpu",weights_only=True);model.load_state_dict(state["model"],strict=True);opt.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"]);start=state["epoch"]+1;best=state["best_top1"]
    try:
        for epoch in range(start,int(cfg["epochs"])):
            loss=train_one_epoch(model,train_loader,opt,a.device,ctx,scaler,max(1,int(cfg["effective_batch"])//int(cfg["microbatch"])),cfg["gradient_clip"]);metrics=validate(model,val_loader,a.device,ctx);row={"epoch":epoch,"train_loss":loss,**metrics,"lr":opt.param_groups[0]["lr"]};append_metrics(run/"metrics.csv",row)
            scheduler.step();state={"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),"epoch":epoch,"best_top1":max(best,metrics["top1"]),"model_key":a.model};torch.save(state,last)
            if metrics["top1"]>best:
                best=metrics["top1"];torch.save(state,run/"best.pth");(run/"best_metadata.json").write_text(json.dumps({"epoch":epoch,"top1":best,"selection_rule":"highest_val_top1_then_earlier_epoch"},indent=2))
            print(row)
    except BaseException as exc:
        (run/"failure.json").write_text(json.dumps({"type":type(exc).__name__,"error":str(exc)},indent=2));raise
    (run/"complete.json").write_text(json.dumps({"status":"completed","best_top1":best},indent=2))
    print(run)
if __name__=="__main__":main()
