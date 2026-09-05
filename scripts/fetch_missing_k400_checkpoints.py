#!/usr/bin/env python3
"""Fetch only explicitly recorded official artifacts, with resume/no-overwrite."""
import argparse, csv, hashlib, urllib.request
from datetime import date
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.registry import ROOT, model_spec, resolve_models

GOOGLE_DRIVE_IDS={"mvit-v1-b-16x4":"194gJinVejq6A1FmySNKQ8vAN5-FOY-QL","dualformer-s":"1dNHPnSl39Vbh7Z-KSWYjNV3PGFlWL3TY"}

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def update_manifest(spec, target, actual):
    path=ROOT/"checkpoints/manifest.csv";fields="path bytes sha256 model_family dataset_hint artifact_role source_url google_drive_id source_project license redistribution_status notes".split()
    rows=list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []
    rel=target.relative_to(ROOT).as_posix(); row=next((r for r in rows if r.get("path","").replace("\\","/")==rel),None)
    if row is None: row={field:"" for field in fields};rows.append(row)
    row.update(path=rel,bytes=str(target.stat().st_size),sha256=actual,model_family=spec["family"],dataset_hint="kinetics400",artifact_role="model-checkpoint",source_url=spec.get("checkpoint_origin","") if str(spec.get("checkpoint_origin","")).startswith("http") else "",google_drive_id=GOOGLE_DRIVE_IDS.get(spec["key"],""),source_project=spec["source_repo"],redistribution_status="unverified-do-not-redistribute",notes=f"Fetched from frozen17 official source and verified {date.today().isoformat()}")
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k,"") for k in fields} for r in rows)

def main():
    p=argparse.ArgumentParser();p.add_argument("--models",nargs="+",default=["frozen17"]);a=p.parse_args(); keys=resolve_models("frozen17" if a.models==["frozen17"] else a.models)
    for key in keys:
        spec=model_spec(key); rel=spec.get("k400_checkpoint")
        if spec["checkpoint_status"].startswith("BLOCKED_EXACT"): print(f"SKIP {key}: exact checkpoint is scientifically blocked");continue
        if not rel: print(f"SKIP {key}: no exact official destination/source");continue
        target=ROOT/rel
        if target.exists(): print(f"EXISTS {key}: {target} sha256={digest(target)}");continue
        target.parent.mkdir(parents=True,exist_ok=True); part=target.with_suffix(target.suffix+".part")
        if key in GOOGLE_DRIVE_IDS:
            try: import gdown
            except ImportError: raise SystemExit("Install requirements-artifacts.txt for official Google Drive downloads")
            gdown.download(id=GOOGLE_DRIVE_IDS[key],output=str(part),resume=True,quiet=False)
        else:
            url=str(spec.get("checkpoint_origin", ""))
            if not url.startswith(("https://github.com/","https://dl.fbaipublicfiles.com/","https://www.dropbox.com/","https://huggingface.co/")):
                print(f"SKIP {key}: no allow-listed official direct URL");continue
            request=urllib.request.Request(url,headers={"Range":f"bytes={part.stat().st_size}-"} if part.exists() else {})
            mode="ab" if part.exists() else "wb"
            with urllib.request.urlopen(request) as response, part.open(mode) as out:
                while block:=response.read(8*1024*1024):out.write(block)
        if not part.exists() or part.stat().st_size==0: raise RuntimeError(f"Empty download for {key}")
        expected=spec.get("checkpoint_sha256"); actual=digest(part)
        if expected and actual!=expected: raise RuntimeError(f"SHA256 mismatch for {key}: {actual}")
        part.replace(target); update_manifest(spec,target,actual); print(f"FETCHED {key}: sha256={actual}")
    print("Run scripts/audit_checkpoints.py --models frozen17 to validate and refresh reports.")
if __name__=="__main__":main()
