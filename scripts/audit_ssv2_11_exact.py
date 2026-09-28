"""Read-only checkpoint/dataset audit; writes only this task's new evidence files."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/ssv2_11_exact_accuracy'
KEYS = ['dualformer-t', 'video-focalnet-t', 'videoswin-t', 'mvit-v1-b-16x4',
        'video-focalnet-s', 'mvit-v1-b-32x3', 'dualformer-s', 'videoswin-s',
        'zeroi2v-b16-8f', 'dualformer-b-in21k', 'omnivore-b-in21k']
EXPECTED = 'c0b8d5fc57ced12f57fb50468e281cae9d98b5a16e15e79be37de8de2c0c8c02'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_name(path.name + '.' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.bak')
        backup.write_bytes(path.read_bytes())
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def inspect_checkpoint(path):
    import torch
    item = {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}
    try:
        if 'ZeroI2V' in path.parts or path.name.startswith('CLIP-'):
            import numpy as np
            from mmengine.logging.history_buffer import HistoryBuffer
            # Explicit metadata types from the author checkpoint; never unrestricted pickle.
            torch.serialization.add_safe_globals([HistoryBuffer, np.core.multiarray.scalar,
                np.core.multiarray._reconstruct, np.dtype, np.ndarray,
                type(np.dtype('float64')), type(np.dtype('float32')), type(np.dtype('int64'))])
        try:
            payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        except RuntimeError as exc:
            if 'mmap' not in str(exc):
                raise
            payload = torch.load(path, map_location='cpu', weights_only=True)
        for wrapper in ('state_dict', 'model_state', 'model', 'module'):
            if isinstance(payload, dict) and isinstance(payload.get(wrapper), dict):
                payload = payload[wrapper]
                item['wrapper'] = wrapper
                break
        tensors = {}
        def flatten(obj, prefix=''):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if torch.is_tensor(v):
                        tensors[prefix+str(k)] = v
                    elif isinstance(v, dict) and k in ('trunk', 'heads'):
                        flatten(v, prefix+str(k)+'.')
        flatten(payload)
        suffixes = ('head.weight', 'head.proj.weight', 'cls_head.fc_cls.weight', 'video.1.weight', 'classifier.weight')
        item['head_shapes'] = {k: list(v.shape) for k,v in tensors.items() if k.endswith(suffixes)}
        item['architecture_shapes'] = {k:list(v.shape) for k,v in tensors.items()
                                       if any(s in k for s in ('patch_embed.proj.weight','patch_embeds.0.proj.weight','conv1.weight','positional_embedding'))}
        item['tensor_count'] = len(tensors)
        item['shape_signature_sha256'] = hashlib.sha256(json.dumps({k:list(v.shape) for k,v in sorted(tensors.items())}, sort_keys=True).encode()).hexdigest()
        item['error'] = ''
    except Exception as exc:
        item['error'] = f'{type(exc).__name__}: {exc}'
    return item

def audit():
    import torch
    import yaml
    sys.path.insert(0, str(ROOT))
    from data import load_ssv2_annotation_map
    specs = yaml.safe_load((ROOT/'configs/frozen17.yaml').read_text(encoding='utf-8'))['models']
    wanted = {s['key']:s for s in specs if s['key'] in KEYS}
    folders = ('dualformer','focalnet','videoswin','mvit','omnivore','omnimae','ZeroI2V','legacy_ssv2_released')
    candidates = set()
    for directory in folders:
        candidates.update(p for p in (ROOT/'checkpoints'/directory).rglob('*') if p.suffix in ('.pth','.pyth','.torch','.pt'))
    for directory in ('models','configs','training','results'):
        candidates.update(p for p in (ROOT/directory).rglob('*') if p.suffix in ('.pth','.pyth','.torch','.pt','.safetensors'))
    previous_path = OUT/'local_checkpoint_inventory.json'
    previous = {r['path']:r for r in json.loads(previous_path.read_text(encoding='utf-8'))} if previous_path.exists() else {}
    inventory = []
    for path in sorted(candidates):
        old = previous.get(path.relative_to(ROOT).as_posix())
        row = old if old and not old['error'] and old.get('head_shapes') and old['bytes'] == path.stat().st_size else inspect_checkpoint(path)
        inventory.append(row)
        print(row['path'], row.get('head_shapes',{}), row['error'][:180], flush=True)
    save(OUT/'local_checkpoint_inventory.json', inventory)
    manifest = ROOT/'manifests/ssv2_1000_seed0.csv'
    clips = list(csv.DictReader(manifest.open(encoding='utf-8-sig')))
    annotations = ROOT/'third_party/UniFormer/video_classification/data_list/sthv2/somesomev2_rgb_validation_split.txt'
    labels_path = ROOT/'datasets/ssv2/labels/labels.json'
    labels = json.loads(labels_path.read_text(encoding='utf-8-sig'))
    mapping = load_ssv2_annotation_map(annotations)
    videos = ROOT/'datasets/ssv2/videos'
    missing = [r['RelativePath'] for r in clips if not (videos/r['RelativePath']).is_file()]
    unresolved = [r['RelativePath'] for r in clips if Path(r['RelativePath']).stem not in mapping]
    # The official file has unquoted commas in labels: only the first comma separates the ID.
    focal_labels = [dict(zip(('id','name'),line.split(',',1))) for line in
                    (ROOT/'third_party/Video-FocalNets/labels/ssv2_174_labels.csv').read_text(encoding='utf-8-sig').splitlines()[1:] if line]
    label_mismatches = [r for r in focal_labels if int(labels.get(r['name'], -1)) != int(r['id'])]
    present_val = sum((videos/(str(k)+'.webm')).is_file() for k in mapping)
    dataset = dict(manifest=manifest.relative_to(ROOT).as_posix(), sha256=sha(manifest), expected_sha256=EXPECTED,
                   clips=len(clips), unique_clips=len({r['RelativePath'] for r in clips}), missing_files=missing,
                   unresolved_labels=unresolved, datasets=sorted({r['Dataset'] for r in clips}),
                   label_count=len(labels), label_ids=list(sorted(int(v) for v in labels.values())),
                   annotation_path=annotations.relative_to(ROOT).as_posix(), annotation_sha256=sha(annotations),
                   label_map_path=labels_path.relative_to(ROOT).as_posix(), label_map_sha256=sha(labels_path),
                   official_focalnet_label_mismatches=label_mismatches,
                   full_validation_count=len(mapping), full_validation_present=present_val,
                   full_validation_missing=len(mapping)-present_val,
                   manifest_labels_blank=sum(not r['Label'] for r in clips),
                   label_source='SSV2 annotation mapping; never K400 labels')
    save(OUT/'dataset_validation.json', dataset)
    env = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python=sys.version, executable=sys.executable,
               platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda,
               cuda_available=torch.cuda.is_available(), device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
               git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
               modules={k:importlib.util.find_spec(k) is not None for k in ('mmcv','mmengine','mmaction','timm','pynvml')})
    save(OUT/'environment.json', env)
    save(OUT/'frozen_specs.json', wanted)
    print(json.dumps({k:v for k,v in dataset.items() if k != 'label_ids'},indent=2),flush=True)

def fetch_sources(extra=False):
    import requests
    from concurrent.futures import ThreadPoolExecutor
    repos = ['sail-sg/dualformer','TalalWasim/Video-FocalNets','SwinTransformer/Video-Swin-Transformer',
             'facebookresearch/SlowFast','MCG-NJU/ZeroI2V','facebookresearch/omnivore']
    urls = {}
    for repo in repos:
        slug = repo.split('/')[-1]
        urls[slug+'_repo'] = f'https://api.github.com/repos/{repo}'
        urls[slug+'_releases'] = f'https://api.github.com/repos/{repo}/releases?per_page=100'
        urls[slug+'_issues'] = f'https://api.github.com/search/issues?q=repo:{repo}+ssv2+OR+sthv2+OR+something&per_page=100'
        urls[slug+'_tree'] = f'https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1'
    urls.update({
        'mvit_zoo':'https://raw.githubusercontent.com/facebookresearch/SlowFast/main/MODEL_ZOO.md',
        'swin_zoo':'https://raw.githubusercontent.com/SwinTransformer/Video-Swin-Transformer/master/README.md',
        'swin_storage_releases':'https://api.github.com/repos/SwinTransformer/storage/releases?per_page=100',
        'focal_zoo':'https://raw.githubusercontent.com/TalalWasim/Video-FocalNets/main/README.md',
        'dual_zoo':'https://raw.githubusercontent.com/sail-sg/dualformer/main/README.md',
        'zero_zoo':'https://raw.githubusercontent.com/MCG-NJU/ZeroI2V/main/README.md',
        'zero_hf':'https://huggingface.co/api/models/MCG-NJU/ZeroI2V/tree/main?recursive=true&expand=false',
        'omnivore_zoo':'https://raw.githubusercontent.com/facebookresearch/omnivore/main/omnivore/README.md',
        'mmaction_swin':'https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/recognition/swin/README.md',
        'mmaction_mvit':'https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/recognition/mvit/README.md',
        'mvit_paper':'https://arxiv.org/pdf/2104.11227',
        'omnivore_paper':'https://openaccess.thecvf.com/content/CVPR2022/papers/Girdhar_Omnivore_A_Single_Model_for_Many_Visual_Modalities_CVPR_2022_paper.pdf',
        'zero_paper':'https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11109.pdf',
        'dual_paper':'https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940566.pdf',
        'focal_paper':'https://openaccess.thecvf.com/content/ICCV2023/papers/Wasim_Video-FocalNets_Spatio-Temporal_Focal_Modulation_for_Video_Action_Recognition_ICCV_2023_paper.pdf',
        'swin_paper':'https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Video_Swin_Transformer_CVPR_2022_paper.pdf',
    })
    for query in ('dualformer','video-focalnet','mvit','omnivore','ZeroI2V','swin sthv2','swin ssv2'):
        urls['hf_search_'+query.replace(' ','_')] = 'https://huggingface.co/api/models?search='+requests.utils.quote(query)+'&limit=100'
    if extra:
        urls = {'dual_paper_corrected':'https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940566.pdf',
                'focal_mirror':'https://raw.githubusercontent.com/innat/Video-FocalNets/main/MODEL_ZOO.md',
                'omnivore_hf_mirror':'https://huggingface.co/api/models/akhaliq/Omnivore/tree/main?recursive=true',
                'swin_hf_mirror':'https://huggingface.co/api/models/Tonic/video-swin-transformer/tree/main?recursive=true',
                'zero_supplement':'https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11109-supp.pdf',
                'omnivore_supplement':'https://openaccess.thecvf.com/content/CVPR2022/supplemental/Girdhar_Omnivore_A_Single_CVPR_2022_supplemental.pdf'}
        for repo,number in [('facebookresearch/SlowFast',433),('facebookresearch/SlowFast',532),
                            ('SwinTransformer/Video-Swin-Transformer',95),('SwinTransformer/Video-Swin-Transformer',35),
                            ('SwinTransformer/Video-Swin-Transformer',78)]:
            name=repo.split('/')[-1]+'_'+str(number)
            urls[name]=f'https://api.github.com/repos/{repo}/issues/{number}'
            urls[name+'_comments']=f'https://api.github.com/repos/{repo}/issues/{number}/comments?per_page=100'
        for repo in ('MCG-NJU/ZeroI2V','facebookresearch/omnivore','TalalWasim/Video-FocalNets','sail-sg/dualformer'):
            urls[repo.split('/')[-1]+'_all_issues']=f'https://api.github.com/repos/{repo}/issues?state=all&per_page=100'
    dest = OUT/'sources'; dest.mkdir(parents=True,exist_ok=True)
    def fetch(pair):
        name,url = pair
        info = dict(name=name,url=url,retrieved_utc=datetime.now(timezone.utc).isoformat())
        try:
            response = requests.get(url,timeout=35,headers={'User-Agent':'SSV2-checkpoint-audit'})
            info.update(status=response.status_code, final_url=response.url)
            suffix = '.pdf' if response.content.startswith(b'%PDF') else '.txt'
            path = dest/(name+suffix)
            path.write_bytes(response.content)
            info.update(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=len(response.content))
        except requests.RequestException as exc:
            info['error']=str(exc)
        print(name,info.get('status'),info.get('bytes'),flush=True)
        return info
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(fetch,urls.items()))
    save(OUT/('source_retrieval_extra.json' if extra else 'source_retrieval.json'), records)

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--fetch-sources',action='store_true'); p.add_argument('--extra',action='store_true'); a=p.parse_args()
    fetch_sources(a.extra) if a.fetch_sources else audit()
