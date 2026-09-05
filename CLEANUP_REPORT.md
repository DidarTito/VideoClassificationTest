# Cleanup report

Audit date: 2026-08-20

No files were archived or deleted during this audit. The classifications below
are deliberately conservative because the RTX 3050 baseline, exact sample,
checkpoints, and model adapters must remain reproducible for the RTX 5070 run.

## Current Stage 2 architecture

`run_benchmark.py` is the benchmark engine and intended central entry point. It
loads clips through `data.py`, constructs models lazily from
`models/__init__.py`, measures power through `power.py`, computes scores through
`metrics.py`, resolves and validates hardware through `utils/device.py`, and
writes each model result atomically. `models/__init__.py` now owns the exact
dataset/checkpoint/preprocessing metadata for all requested pairs.
`stage2_pipeline.py` is an older
orchestrator around the same engine that also emits Stage 1 screening tables,
rankings, a plot, and a report. `check_models.py` and
`check_dataset_models.py` provide the current read-only asset preflight.

The active model registry exposes the 17 selected Stage 2 keys plus several
older, independently useful adapters. Third-party model sources are vendored
under `third_party/`; they should be treated as dependency/provenance trees,
not pruned file by file.

## Repository and artifact layout

- `datasets/kinetics400/`: 11,822 files, about 24.3 GiB. The exact 1,000-clip
  cache is
  `datasets/kinetics400/.clip_cache_v4_32f_224px_1000c_seed0_all/`.
- `datasets/ssv2/`: 114,455 files, about 13.9 GiB, including labels, videos,
  and split-aware clip caches.
- `checkpoints/`: 46 inventoried artifacts, about 21.1 GiB. Binary artifacts
  are ignored by Git; `checkpoints/manifest.csv` records paths, sizes, hashes,
  provenance, and dataset hints.
- `results/rtx3050/`: reviewed historical results. The K400 metrics file has
  15 successful model rows and the SSV2 metrics file has 6. These files must
  remain untouched.
- `data/stage1_candidates_k400.csv` and
  `data/stage2_candidate_availability.csv`: small, tracked research manifests
  used by Stage 1 and cross-dataset reporting.
- There was no tracked top-level `manifests/` directory at initial audit time.
  The ignored cache orders have now been promoted without resampling to
  `manifests/k400_1000_seed0.csv` (SHA-256
  `b8ec0d92f50b4ae5dff46754f62c2ef1b2b8467332f88bb0cc29c8e6172ff749`)
  and `manifests/ssv2_1000_seed0.csv` (SHA-256
  `c0b8d5fc57ced12f57fb50468e281cae9d98b5a16e15e79be37de8de2c0c8c02`).
  Each contains exactly 1,000 ordered source-video rows.

Existing exact cache manifests at audit time:

| Dataset | Entries | Cache manifest SHA-256 |
|---|---:|---|
| K400 | 1,000 | `acde32345187fe23a2bd9c985b66bd6dacc8e8b2b3f4ce15052c86c55a6b3d34` |
| SSV2 | 1,000 | `9b8950da5c12f05a43a321c230baa96187c5430931fd56a1a1420fe41c0664ef` |

Both manifests referenced 1,000 present source videos during the audit. Their
source order is part of the experiment and must not be regenerated.

## KEEP / ACTIVE

### Benchmark and data path

- `run_benchmark.py`, `data.py`, `metrics.py`, and `power.py`: active Stage 2
  engine.
- `utils/device.py`: active central device resolution, CUDA-kernel
  compatibility validation, metadata, safe output naming, and temperature
  support.
- `manifests/k400_1000_seed0.csv` and `manifests/ssv2_1000_seed0.csv`: active
  portable sample identities; do not regenerate for a cross-device run.
- `models/__init__.py`, `models/common.py`, and
  `models/kinetics_labels.py`: active registry and shared adapter support.
- `models/uniformer.py`, `models/mvit_slowfast.py`, `models/videoswin.py`,
  `models/timesformer.py`, `models/videofocalnet.py`, `models/dualformer.py`,
  `models/omnivore.py`, `models/videomae.py`, and `models/vtn.py`: active exact
  K400 model adapters.
- `models/vivit.py` and `models/svt.py`: active scientific safeguards that
  reject unavailable/non-classifier checkpoint substitutions.
- `models/actionclip.py`, `models/mvit.py`, and `models/zeroi2v.py`: not in the
  selected 15-model K400 run, but still exposed by the public registry; keep
  until the registry is intentionally versioned.
- `check_models.py` and `check_dataset_models.py`: current preflight logic and
  test dependencies. Keep until the central `--preflight` path demonstrably
  covers the same checks.
- `screen_models.py`: active Stage 1 compatibility module used by
  `stage2_pipeline.py`.

### Operational and research support

- `scripts/checkpoint_artifacts.py`: active checkpoint inventory/verification
  utility.
- `scripts/plot_stage1_screening.py`: canonical Stage 1 figure generator.
- `scripts/summarize_stage2_results.py`: canonical K400/SSV2 comparison tool.
- `scripts/run_requested_models.ps1`: current exact-model launcher; keep until
  `scripts/run_k400_device.ps1` and the central CLI complete a real smoke run.
- `scripts/smoke_test_ssv2_models.py`: SSV2-specific adapter check; keep until
  equivalent SSV2 coverage exists in the central smoke mode.
- All files under `tests/`: active, fast contract coverage.
- `requirements.txt`, `requirements-lock.txt`,
  `requirements-device-metrics.txt`, `requirements-artifacts.txt`,
  `setup_local_packages.ps1`, and `environment-info.txt`: environment records
  and setup inputs. `environment-info.txt` is a historical RTX 3050 snapshot,
  not runtime configuration.
- `README.md`, `RUNNING_STAGE1_STAGE2.md`,
  `STAGE1_THRESHOLD_RATIONALES.md`, `STAGE2_NETSCORE_RESULTS.md`,
  `THIRD_PARTY.md`, and every third-party README/license: research history,
  methodology, provenance, and operating documentation.
- `data/`, `datasets/`, `checkpoints/`, `results/`, `figures/`,
  `third_party/`, `.git/`, `.gitattributes`, and `.gitignore`: retain.
- `venv/`: ignored and recreatable, but keep locally until the RTX 5070 run is
  complete because it is the currently exercised environment.

## ARCHIVE

These are candidates for `archive/legacy/` only after the new central workflow
passes preflight, smoke, resume, and output-contract tests. No move was made.

- `stage2_pipeline.py`: older all-in-one Stage 2 wrapper that overlaps the new
  central benchmark CLI, but still provides screening/report generation.
- `scripts/run_requested_models.ps1`: superseded only after the new device
  runner proves feature parity.
- `scripts/add_netscore_variants.py`: legacy SAM-era CSV retrofit tool. It is
  useful for historical data but should not be in the new result-generation
  path.
- `scripts/smoke_test_ssv2_models.py`: specialized predecessor to central
  smoke mode; retain until SSV2 parity is tested.
- `make_test_video.py`: one-off synthetic video generator with no imports or
  documentation references.
- `scripts/download_hf_subset.py`: experimental dataset downloader; it samples
  without the experiment seed/manifest protocol and is not imported anywhere.
- `scripts/download_kinetics_subset.py`: experimental downloader whose
  `process_split` function is never invoked; it is not part of Stage 2.
- `models/motionformer.py`: unregistered placeholder for a missing source tree,
  not an exact working model-checkpoint pair.
- `paretto_all.py`: misspelled backward-compatible wrapper around the canonical
  Stage 1 plot script.
- `pareto_plot.png` and `stage1_pareto_plot.png`: older root-level generated
  plots; the canonical current PNG/SVG/PDF set is under `figures/`.
- `benchmark_results.csv`: legacy SAM-bearing output. Preserve it unchanged as
  historical data; archive classification does not authorize editing or
  deletion.

`RUNNING_STAGE1_STAGE2.md` and `STAGE2_NETSCORE_RESULTS.md` also contain older
commands/folder names. Prefer updating or explicitly labeling them historical
instead of deleting them.

## DELETE_SAFE

Only generated, reproducible debris is safe for direct deletion, and none was
deleted in this pass:

- `__pycache__/`, `models/__pycache__/`, `scripts/__pycache__/`,
  `tests/__pycache__/`, third-party `__pycache__/` directories, and `*.pyc`.
- `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, and coverage output if they
  appear.
- The fully commented obsolete implementation block at the start of
  `screen_models.py`, after confirming no documentation tool extracts it.
- `models/missing_models.py`, which contains only a deprecation docstring and
  has no imports or registry entry, after one final import scan.

Incomplete video-download `.part`/`.ytdl` files and old clip-cache versions
look generated, but they are inside `datasets/`. They are intentionally not
classified for immediate deletion because the repository cleanup rules say
not to delete dataset material and tomorrow's reproducibility has priority.

## Existing tests and focused strategy

Baseline command:

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -v
```

At audit time all 71 original tests passed. After the portability work, all
114 tests pass in about 5 seconds. Tests now cover
deployment NetScore formulas, output/logit contracts, energy units, power
logger cleanup, NVML backend selection, label resolution, split-aware cached
sampling, checkpoint manifests, exact registry variants, Stage 1 screening,
cross-dataset summaries, central device resolution, manifest stability,
dynamic output naming, CLI parsing, resume/retry semantics, failure isolation,
and the CPU/NVML-unavailable path.

The focused portability tests use CPU/mocks rather than loading checkpoints:

1. `tests/test_device.py`: `auto`, explicit CPU/CUDA/index parsing, Blackwell
   kernel failure, metadata, UUID-aware NVML mapping, GPU-name sanitization,
   and genuine CPU fallback.
2. `tests/test_manifest.py`: exact ordered clip IDs, SHA-256 stability,
   scientific metadata, path safety, and exact manifest/cache alignment.
3. `tests/test_run_configuration.py`: CLI defaults/model selection, dynamic
   output names, complete CSV schema, strict resume identity, failed-row retry,
   CPU behavior, and model failure isolation.
4. `tests/test_power.py` and `tests/test_benchmark_contract.py`: direct NVML,
   selected-device propagation, nvidia-smi fallback, logger cleanup, missing
   power behavior, energy units, and strict dataset-sized logits.
5. `tests/test_requested_model_metadata.py`: all 17 requested candidates,
   exact dataset/checkpoint/preprocessing identities, explicit unavailability,
   and separation of Stage 1 screening from checkpoint-pair publications.
6. `tests/test_metrics.py`: the unchanged deployment NetScore formulas and
   their percent/seconds/watts/MiB input contract.

Final hardware verification should then be intentionally small and ordered:

1. Run central preflight and require CUDA kernel compatibility, dataset,
   manifest, checkpoint, import, and NVML reports.
2. Run UniFormer-S on 2-5 manifest clips through the complete CSV path.
3. Run a second architecturally different model on the same clips.
4. Run the 100-clip quick pass, inspect the extrapolated runtime, then start the
   1,000-clip resumable run only if the estimate fits the available window.
