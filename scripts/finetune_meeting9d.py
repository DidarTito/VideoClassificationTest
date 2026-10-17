#!/usr/bin/env python3
import csv,gc,hashlib,json,math,os,random,subprocess,sys,time,traceback
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.adapters.runtime import build_trainable
from training.amp import amp_policy,autocast
from training.engine import train_one_epoch
from training.registry import model_spec
from training.validation import validate
from scripts.finetune_ssv2_full import FullFTDataset,collate_skip_bad,build_splits,build_scheduler

OUT=Path("/media/esai01/10A2D9F6A2D9DFF0/ssv2_meeting9d")
COMMON=OUT/"_common"; PREF=OUT/"preflight.json"; STATUS=OUT/"queue_status.json"
COMMIT="9cdc8cae57116964000339df9450e2fe9c8025f8"
HOLD_SHA="c0b8d5fc57ced12f57fb50468e281cae9d98b5a16e15e79be37de8de2c0c8c02"
PLANS={
"video-focalnet-t":(30,12,5),
"mvit-v1-b-16x4":(30,12,5),
"video-focalnet-s":(30,12,5),
"videoswin-t":(30,12,5),
"dualformer-t":(30,12,5),
"mvit-v1-b-32x3":(30,12,5),
"videoswin-s":(30,12,5),
"dualformer-b-in21k":(30,12,5),
"omnivore-b-in21k":(30,12,5)}
ORDER=[
"video-focalnet-t",
"mvit-v1-b-16x4",
"video-focalnet-s",
"videoswin-t",
"dualformer-t",
"mvit-v1-b-32x3",
"videoswin-s",
"dualformer-b-in21k",
"omnivore-b-in21k"]
MILE={10,20,30}
SEED=20260928; EFF=16; BLR=3e-5; HLR=3e-4; WD=.05; SMOOTH=.1; WARM=3; CLIP=1.; WORKERS=8; DELTA=.15
def jdump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp")
 q.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n");os.replace(q,p)
def digest(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def commit():return subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
def seed():random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
def cpu_state(m):return {k:v.detach().cpu() for k,v in m.state_dict().items()}
def dotted(o,path):
 for x in path.split("."):o=o[int(x)] if x.isdigit() else getattr(o,x)
 return o
def head_of(model,owner,path):
 # First try the exact registry path directly.
 for label,o in (("owner",owner),("model",model)):
  if o is None:
   continue
  try:
   h=dotted(o,path)
   if isinstance(h,nn.Module):
    print(f"[HEAD] {path} -> {label}.{path}",flush=True)
    return h
  except Exception:
   pass

 # build_trainable() wraps the real backend model, so search all
 # registered nested nn.Modules for the exact registry-path suffix.
 roots=[]
 seen=set()

 def add_root(label,obj):
  if isinstance(obj,nn.Module) and id(obj) not in seen:
   seen.add(id(obj))
   roots.append((label,obj))

 add_root("model",model)
 add_root("owner",owner)

 # Some owners are ordinary wrapper objects whose attributes contain
 # the actual nn.Module.
 for label,obj in (("model",model),("owner",owner)):
  if obj is None:
   continue
  try:
   for name,v in vars(obj).items():
    if isinstance(v,nn.Module):
     add_root(f"{label}.{name}",v)
  except Exception:
   pass

 found=[]
 for root_name,root in roots:
  for name,mod in root.named_modules():
   if name == path or name.endswith("." + path):
    found.append((root_name,name,mod))

 # Deduplicate the same module reached through multiple wrapper roots.
 unique=[]
 seen_mod=set()
 for root_name,name,mod in found:
  if id(mod) not in seen_mod:
   seen_mod.add(id(mod))
   unique.append((root_name,name,mod))

 if len(unique) == 1:
  root_name,name,mod=unique[0]
  print(f"[HEAD] {path} -> {root_name}.{name}",flush=True)
  return mod

 if len(unique) > 1:
  raise RuntimeError(
   f"ambiguous classifier {path}: " +
   ", ".join(f"{r}.{n}" for r,n,_ in unique)
  )

 raise RuntimeError(
  f"classifier not found recursively: {path}; "
  f"searched roots={[x[0] for x in roots]}"
 )
def pair_collate(batch,key):
 z=collate_skip_bad(batch)
 if z is None:
  return None

 clips,labels=z[0],z[1]

 # Validated FullFTDataset returns uint8 B,T,C,H,W in [0,255].
 # TrainableModel performs its own mean/std normalization, therefore
 # only convert to float [0,1] here -- DO NOT normalize a second time.
 clips=clips.to(torch.float32).div_(255.0)

 spec=model_spec(key)
 target_frames=int(spec["input"]["num_frames"])

 if clips.shape[1] != target_frames:
  idx=torch.linspace(
   0,
   clips.shape[1]-1,
   target_frames
  ).long()

  clips=clips.index_select(1,idx)

 # Contract entering TrainableModel:
 # B,T,C,H,W float32 [0,1], correct T for this architecture.
 if clips.ndim != 5 or clips.shape[2] != 3:
  raise RuntimeError(
   f"{key}: bad prepared input shape {tuple(clips.shape)}"
  )

 if clips.shape[1] != target_frames:
  raise RuntimeError(
   f"{key}: expected {target_frames} frames, "
   f"got {clips.shape[1]}"
  )

 if not torch.isfinite(clips).all():
  raise RuntimeError(f"{key}: prepared input contains non-finite values")

 lo=float(clips.min())
 hi=float(clips.max())

 if lo < -1e-6 or hi > 1.000001:
  raise RuntimeError(
   f"{key}: prepared input outside [0,1]: min={lo} max={hi}"
  )

 return clips,labels


WORKERS=8
def loader(key,ds,bs,shuffle,workers=WORKERS):
 from functools import partial

 kw=dict(
  batch_size=bs,
  shuffle=shuffle,
  num_workers=workers,
  pin_memory=True,
  persistent_workers=workers>0,
  collate_fn=partial(pair_collate,key=key),
  drop_last=False,
 )

 if workers:
  kw["prefetch_factor"]=2

 return DataLoader(ds,**kw)

def build(key):
 s=model_spec(key); src=ROOT/s["k400_checkpoint"]
 if s.get("checkpoint_status")!="READY_EXACT" or not src.is_file():raise RuntimeError(f"{key}: exact K400 source unavailable")
 sh=digest(src)
 if sh!=s["checkpoint_sha256"]:raise RuntimeError(f"{key}: source SHA mismatch")
 m,o=build_trainable(key,"cuda:0");m=m.cuda()
 for p in m.parameters():p.requires_grad_(True)
 if isinstance(o,nn.Module):
  for p in o.parameters():p.requires_grad_(True)
 h=head_of(m,o,s["classifier"]["path"]);hid={id(p) for p in h.parameters()}
 ps=list(m.parameters()); mids={id(p) for p in ps}
 if not hid or not hid<=mids:raise RuntimeError(f"{key}: head params not registered")
 bp=[p for p in ps if id(p) not in hid];hp=list(h.parameters())
 opt=torch.optim.AdamW([{"params":bp,"lr":BLR},{"params":hp,"lr":HLR}],weight_decay=WD)
 ids=[id(p) for g in opt.param_groups for p in g["params"]]
 total=sum(p.numel() for p in ps);trainable=sum(p.numel() for p in ps if p.requires_grad)
 if total!=trainable or set(ids)!=mids or len(ids)!=len(set(ids)):raise RuntimeError(f"{key}: optimizer/trainability audit failed")
 dtype=torch.bfloat16;scaler=None
 return m,o,hid,opt,dtype,scaler,dict(source=str(src),source_sha256=sh,total_params=total,trainable_params=trainable,classifier=s["classifier"]["path"])

def probe(key, train):
    errs = []

    for bs in (16, 8, 4, 2, 1):
        m = o = opt = scaler = None

        try:
            print(f"[PRECHECK] {key} batch={bs}", flush=True)
            seed()

            m, o, hid, opt, dtype, scaler, meta = build(key)

            # Sample backbone tensors from throughout the network,
            # not only the first few parameters.
            backbone = [
                (n, p)
                for n, p in m.named_parameters()
                if id(p) not in hid
                and p.requires_grad
                and p.numel() > 0
            ]

            if not backbone:
                raise RuntimeError("no trainable backbone parameters found")

            idx = sorted(set([
                0,
                1,
                2,
                len(backbone) // 4,
                len(backbone) // 2,
                (3 * len(backbone)) // 4,
                len(backbone) - 3,
                len(backbone) - 2,
                len(backbone) - 1,
            ]))

            tracked = [
                backbone[i]
                for i in idx
                if 0 <= i < len(backbone)
            ]

            before = {
                n: p.detach().cpu().clone()
                for n, p in tracked
            }

            dl = loader(
                key,
                Subset(train, range(min(bs, len(train)))),
                bs,
                False,
                2,
            )

            ctx = lambda: autocast("cuda:0", dtype)

            loss = train_one_epoch(
                m,
                dl,
                opt,
                "cuda:0",
                ctx,
                scaler,
                1,
                CLIP,
            )

            changed_names = [
                n
                for n, p in tracked
                if not torch.equal(
                    before[n],
                    p.detach().cpu()
                )
            ]

            changed = len(changed_names)

            if changed == 0:
                raise RuntimeError(
                    "backbone did not update; retry smaller microbatch"
                )

            meta.update(
                status="PASS",
                micro_batch=bs,
                audit_loss=float(loss),
                tracked_backbone_changed=changed,
                changed_examples=changed_names,
                amp_dtype=str(dtype),
            )

            print(
                f"[PASS] {key} batch={bs} "
                f"changed={changed} dtype={dtype}",
                flush=True,
            )

            return meta

        except Exception as e:
            msg = str(e).lower()

            errs.append({
                "batch": bs,
                "error": f"{type(e).__name__}: {e}",
            })

            print(
                f"[FAIL] {key} batch={bs}: {e}",
                flush=True,
            )

            retryable = (
                isinstance(e, torch.cuda.OutOfMemoryError)
                or "out of memory" in msg
                or "backbone did not update" in msg
            )

            if not retryable:
                break

        finally:
            try:
                del m, o, opt, scaler
            except Exception:
                pass

            gc.collect()
            torch.cuda.empty_cache()

    raise RuntimeError(json.dumps(errs))

def save_csv(p,rows):
 if not rows:return
 q=p.with_suffix(".tmp")
 with q.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 os.replace(q,p)
def read_csv(p):
 if not p.is_file():return []
 out=[]
 with p.open(newline="") as f:
  for r in csv.DictReader(f):
   out.append({k:(int(float(v)) if k in {"epoch","best_epoch","stale"} else float(v)) for k,v in r.items()})
 return out
def save_weights(p,m,key,ep,val,sha):
 st=cpu_state(m);torch.save({"format":"ssv2_full_ft_weights_v2","model_key":key,"dataset":"ssv2","num_classes":174,
  "epoch":ep,"val_top1":val,"model":st,"model_state_dict":st,"source_sha256":sha},p)
def plot(out,key,rows):
 try:
  import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
  e=[r["epoch"] for r in rows];best=max(rows,key=lambda r:r["val_top1"])["epoch"]
  fig,ax=plt.subplots();ax.plot(e,[r["train_loss"] for r in rows],label="Train loss");ax.plot(e,[r["val_loss"] for r in rows],label="Selection-val loss");ax.axvline(best,ls="--",label=f"Best epoch {best}");ax.set(xlabel="Epoch",ylabel="Loss",title=f"{key}: loss curve");ax.legend();fig.tight_layout();fig.savefig(out/"loss_curve.png",dpi=180);plt.close(fig)
  fig,ax=plt.subplots();ax.plot(e,[r["val_top1"] for r in rows],marker="o",ms=2,label="Selection-val Top-1");ax.axvline(best,ls="--",label=f"Best epoch {best}");ax.set(xlabel="Epoch",ylabel="Top-1 (%)",title=f"{key}: validation curve");ax.legend();fig.tight_layout();fig.savefig(out/"val_top1_curve.png",dpi=180);plt.close(fig)
 except Exception as e:(out/"plot_error.txt").write_text(str(e))
def plot_all():
 try:
  import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
  fig,ax=plt.subplots();ok=False
  for k in ORDER:
   rows=read_csv(OUT/k/"training_history.csv")
   if rows:ok=True;ax.plot([r["epoch"] for r in rows],[r["val_top1"] for r in rows],label=k)
  if ok:ax.set(xlabel="Epoch",ylabel="Selection-val Top-1 (%)",title="All models: K400 → SSV2 Full FT");ax.legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/"all_models_val_top1.png",dpi=200)
  plt.close(fig)
 except Exception as e:(OUT/"all_models_plot_error.txt").write_text(str(e))

def opt_to(opt,dev):
 for s in opt.state.values():
  for k,v in s.items():
   if torch.is_tensor(v):s[k]=v.to(dev)

def train(key,train_rows,val_rows,pre):
 maxep,minep,pat=PLANS[key];out=OUT/key;ck=out/"checkpoints";ck.mkdir(parents=True,exist_ok=True)
 if (out/"complete.json").is_file():return
 bs=int(pre[key]["micro_batch"]);acc=math.ceil(EFF/bs);seed()
 trds=FullFTDataset(train_rows,train=True);vds=FullFTDataset(val_rows,train=False)
 tr=loader(key,trds,bs,True);va=loader(key,vds,bs,False)
 m,o,hid,opt,dtype,scaler,meta=build(key)
 updates=math.ceil(len(tr)/acc);sch=build_scheduler(opt,updates*maxep,updates*WARM);ctx=lambda:autocast("cuda:0",dtype)
 hist=read_csv(out/"training_history.csv");start=1;best=-1.;be=0;eb=-1.;stale=0
 last=ck/"latest.pt"
 if last.is_file():
  s=torch.load(last,map_location="cpu",weights_only=False)
  if s["max_epochs"]!=maxep or s["source_sha256"]!=meta["source_sha256"]:raise RuntimeError("resume provenance/horizon mismatch")
  m.load_state_dict(s["model"],strict=True);opt.load_state_dict(s["optimizer"]);opt_to(opt,"cuda:0");sch.load_state_dict(s["scheduler"])
  if scaler is not None and s.get("scaler") is not None:scaler.load_state_dict(s["scaler"])
  start=s["epoch"]+1;best=s["best"];be=s["best_epoch"];eb=s["early_best"];stale=s["stale"]
 hook=opt.register_step_post_hook(lambda *_a,**_k:sch.step())
 conf=dict(model=key,max_epochs=maxep,min_epochs=minep,patience=pat,min_delta_pp=DELTA,micro_batch=bs,grad_accum=acc,effective_batch=bs*acc,
  backbone_lr=BLR,head_lr=HLR,weight_decay=WD,label_smoothing=SMOOTH,warmup_epochs=WARM,train_videos=len(train_rows),selection_val_videos=len(val_rows),
  fixed_holdout_videos=1000,fixed_holdout_used_for_selection=False,source=meta,git_commit=commit())
 if key=="dualformer-b-in21k":conf["variant_warning"]="Code key only: exact local K400 Base source corresponds to K400 81.1 / IN1K-pretrained release, not frozen Stage-1 IN21K 82.9 row."
 jdump(out/"config.json",conf)
 stop=f"max_epochs:{maxep}"
 for ep in range(start,maxep+1):
  t=time.time();print(f"\n===== {key} epoch {ep}/{maxep} micro={bs} accum={acc} =====",flush=True)
  tl=train_one_epoch(m,tr,opt,"cuda:0",ctx,scaler,acc,CLIP);vm=validate(m,va,"cuda:0",ctx)
  top=float(vm["top1"]);vl=float(vm.get("loss",float("nan")));v5=float(vm.get("top5",float("nan")))
  if top>best:best=top;be=ep;save_weights(ck/"best_weights.pt",m,key,ep,top,meta["source_sha256"])
  if top>=eb+DELTA:eb=top;stale=0
  else:stale+=1
  row=dict(epoch=ep,train_loss=float(tl),val_loss=vl,val_top1=top,val_top5=v5,backbone_lr=float(opt.param_groups[0]["lr"]),
   head_lr=float(opt.param_groups[-1]["lr"]),best_epoch=be,best_val_top1=best,stale=stale,epoch_seconds=time.time()-t)
  hist=[r for r in hist if r["epoch"]!=ep]+[row];hist.sort(key=lambda r:r["epoch"]);save_csv(out/"training_history.csv",hist);plot(out,key,hist)
  state={"format":"ssv2_full_ft_resume_v2","model_key":key,"epoch":ep,"max_epochs":maxep,"best":best,"best_epoch":be,"early_best":eb,"stale":stale,
   "source_sha256":meta["source_sha256"],"model":cpu_state(m),"optimizer":opt.state_dict(),"scheduler":sch.state_dict(),"scaler":scaler.state_dict() if scaler else None}
  torch.save(state,last)
  if ep in MILE or ep==maxep:save_weights(ck/f"epoch{ep}_weights.pt",m,key,ep,top,meta["source_sha256"])
  print(f"[E{ep}] train_loss={tl:.4f} val_loss={vl:.4f} val_top1={top:.2f}% best={best:.2f}@{be} stale={stale}",flush=True)
  if ep>=minep and stale>=pat:stop=f"early_stop:no >= {DELTA}pp gain for {stale} epochs";break
 last_ep=hist[-1]["epoch"];needs=bool(last_ep>=maxep and be>=maxep-2)
 jdump(out/"complete.json",dict(status="completed",model=key,stop_reason=stop,last_epoch=last_ep,best_epoch=be,best_val_top1=best,needs_extension=needs,fixed_1000_used_for_selection=False))
 del m,o,opt,scaler;gc.collect();torch.cuda.empty_cache()

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if commit()!=COMMIT:raise SystemExit(f"wrong commit {commit()} expected {COMMIT}")
 if not torch.cuda.is_available():raise SystemExit("CUDA unavailable")
 train_rows,val_rows,hold,summary=build_splits(COMMON)
 if len(train_rows)!=168913 or len(val_rows)!=23777 or len(hold)!=1000 or summary["holdout_sha256"]!=HOLD_SHA:raise SystemExit(f"split mismatch {len(train_rows)}/{len(val_rows)}/{len(hold)}")
 ap=__import__("argparse").ArgumentParser();ap.add_argument("--preflight",action="store_true");ap.add_argument("--run",action="store_true");a=ap.parse_args()
 if a.preflight:
  rep={"git_commit":commit(),"models":{}}
  ds=FullFTDataset(train_rows,train=True)
  for k in ORDER:
   try:rep["models"][k]=probe(k,ds)
   except Exception as e:rep["models"][k]={"status":"FAIL","error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc()}
   jdump(PREF,rep)
  passed=[k for k,v in rep["models"].items() if v.get("status")=="PASS"];failed=[k for k,v in rep["models"].items() if v.get("status")!="PASS"];print("PASS:",passed);print("FAIL:",failed);raise SystemExit(0 if not failed else 2)
 if not a.run:raise SystemExit("use --preflight or --run")
 rep=json.loads(PREF.read_text());pre=rep["models"];status={}
 for k in ORDER:
  if pre.get(k,{}).get("status")!="PASS":status[k]={"status":"SKIPPED_PREFLIGHT_FAIL","error":pre.get(k,{}).get("error")};jdump(STATUS,status);continue
  try:status[k]={"status":"RUNNING"};jdump(STATUS,status);train(k,train_rows,val_rows,pre);status[k]=json.loads((OUT/k/"complete.json").read_text())
  except Exception as e:status[k]={"status":"FAILED","error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc()};(OUT/k).mkdir(parents=True,exist_ok=True);jdump(OUT/k/"failure.json",status[k]);gc.collect();torch.cuda.empty_cache()
  jdump(STATUS,status);plot_all()
 print(json.dumps(status,indent=2));plot_all()
if __name__=="__main__":main()
