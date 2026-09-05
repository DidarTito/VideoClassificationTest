#!/usr/bin/env python3
"""Final fine-tuned SSV2 deployment measurement; batch size = 1 is mandatory."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import run_benchmark
from data import load_annotation_ids, load_clip_set, load_ground_truth
from training.adapters.runtime import build_trainable
from training.registry import ROOT, FROZEN17_KEYS, model_spec


class DeploymentAdapter:
    def __init__(self, key, checkpoint, device):
        self.spec=model_spec(key);self.model,self.owner=build_trainable(key,device)
        state=torch.load(checkpoint,map_location="cpu",weights_only=True);self.model.load_state_dict(state["model"],strict=True);self.model.eval();self.device=device
        self.name=self.spec["display_name"];self.accuracy=float("nan");self.gflops=self.spec["stage1"]["gflops_per_view"];self.frames=self.spec["input"]["num_frames"];self.class_names=[str(i) for i in range(174)]
    @property
    def parameters(self):return sum(p.numel() for p in self.model.parameters())
    @torch.inference_mode()
    def __call__(self,clip):return self.model(clip.to(self.device,non_blocking=True))

def main():
    p=argparse.ArgumentParser();p.add_argument("--model",required=True,choices=FROZEN17_KEYS);p.add_argument("--checkpoint",required=True);p.add_argument("--device",default="cuda");p.add_argument("--power",default="nvml");p.add_argument("--num-clips",type=int,default=1000);p.add_argument("--seed",type=int,default=0);p.add_argument("--output",default="results/finetuned/ssv2_rtx5070");a=p.parse_args()
    out=(ROOT/a.output).resolve();safe=(ROOT/"results/finetuned/ssv2_rtx5070").resolve()
    if safe not in (out,*out.parents):raise SystemExit("Final measurements must stay below results/finetuned/ssv2_rtx5070")
    annotations=ROOT/"third_party/UniFormer/video_classification/data_list/sthv2/somesomev2_rgb_validation_split.txt";videos=ROOT/"datasets/ssv2/videos";ids=load_annotation_ids(annotations);clips,paths=load_clip_set(videos,max_clips=a.num_clips,frames=32,size=224,seed=a.seed,allowed_stems=ids);labels=load_ground_truth(paths,"ssv2",annotations)
    adapter=DeploymentAdapter(a.model,a.checkpoint,a.device);run_benchmark.MODEL_REGISTRY={a.model:lambda **_:adapter}
    metrics,errors=run_benchmark.benchmark_models([a.model],clips,a.power,"RTX5070",labels,dataset="ssv2",device=a.device,split_name="validation",annotation_file=str(annotations))
    out.mkdir(parents=True,exist_ok=True);metrics.to_csv(out/f"{a.model}_device_metrics.csv",index=False)
    if errors:raise SystemExit(str(errors))
    print(metrics.to_string(index=False))
if __name__=="__main__":main()
