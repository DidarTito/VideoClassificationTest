TASK: RUN THE 11 REMAINING FROZEN MODELS ON SSV2

Repository:
C:\Users\DIDAR\Documents\VideoClassificationTest

DO NOT rerun these 6 already completed models:
- uniformer-s
- uniformer-b
- videomae-b
- videoswin-b
- timesformer-b
- video-focalnet-b

ONLY run these 11:

1. dualformer-t
2. video-focalnet-t
3. videoswin-t
4. mvit-v1-b-16x4
5. video-focalnet-s
6. mvit-v1-b-32x3
7. dualformer-s
8. videoswin-s
9. zeroi2v-b16-8f
10. dualformer-b-in21k
11. omnivore-b-in21k

==================================================
CHECKPOINT RULE
==================================================

For each model:

1. First look for an EXACT SSV2-trained checkpoint.

If an exact SSV2 checkpoint exists:
- use it
- verify classifier output = 174 classes
- protocol name:
  SSV2_CHECKPOINT
- calculate valid SSV2 Top-1 and Top-5
- AccuracyValid = TRUE

2. If NO exact SSV2 checkpoint exists:
use the exact K400 pretrained checkpoint already used in our project.

Protocol name:
PRETRAIN_K400_ON_SSV2

This means:
- run the K400 pretrained model on the SAME SSV2 video inputs
- measure device performance normally
- DO NOT report SSV2 Top-1/Top-5
- AccuracyValid = FALSE
- MeasuredTop1 = N/A
- MeasuredTop5 = N/A
- NS-E / NS-M / NS# = N/A

A 400-class K400 classifier is NOT valid SSV2 classification accuracy.

DO NOT:
- create random 174-class heads
- map K400 labels to SSV2 labels
- substitute model variants
- fine-tune anything
- copy paper accuracy into measured accuracy

==================================================
DATASET
==================================================

Use:

manifests/ssv2_1000_seed0.csv

Expected SHA256:

c0b8d5fc57ced12f57fb50468e281cae9d98b5a16e15e79be37de8de2c0c8c02

Use exactly the same 1000 SSV2 clips for all models.

Settings:

batch size = 1
precision = FP32
seed = 0
warmup = 2
NVML sampling interval = 20 ms

Use each model's correct:
- frame count
- temporal sampling
- resolution
- normalization

Do not force identical ModelFrames when architectures require different input recipes.

==================================================
MEASURE FOR ALL 11
==================================================

For EVERY model measure:

- Model
- Protocol
- Checkpoint path
- Checkpoint SHA256
- Dataset
- Evaluated clips
- GFLOPs
- Runtime parameters
- ModelFrames
- DecodedFrames
- Mean latency ms
- Latency std ms
- FPS
- Average GPU power W
- Energy per clip J
- Total energy for 1000 clips J
- Peak GPU memory MiB

For SSV2_CHECKPOINT models also measure:

- Top-1 correct
- Top-5 correct
- Measured Top-1 %
- Measured Top-5 %
- AccuracyValid = TRUE

For PRETRAIN_K400_ON_SSV2:

- Top-1 = N/A
- Top-5 = N/A
- AccuracyValid = FALSE

==================================================
NETSCORE
==================================================

ONLY calculate NetScore when AccuracyValid = TRUE.

A = measured SSV2 Top-1 percentage
t = total timed inference seconds
P = average power W
r = peak VRAM MiB

NS-E =
20*log10(A^2 / (t*P)^(1/8))

NS-M =
20*log10(A^2 / r^(1/8))

NS# =
20*log10(A^2 / (r*t*P)^(1/8))

Never calculate NetScore for PRETRAIN_K400_ON_SSV2.

==================================================
SAVE RESULTS
==================================================

Create:

results/ssv2_remaining11/

Save after EVERY completed model so interrupted runs can resume.

Master CSV:

results/ssv2_remaining11/device_metrics.csv

Columns:

Model
Protocol
Checkpoint
CheckpointSHA256
AccuracyValid
MeasuredTop1
MeasuredTop5
GFLOPs
ParamsM
ModelFrames
DecodedFrames
LatencyMeanMs
LatencyStdMs
FPS
AveragePowerW
EnergyPerClipJ
TotalEnergyJ
PeakVRAMMiB
NS-E
NS-M
NS#
Status
Notes

Also create:

reports/SSV2_REMAINING11_RESULTS.md

==================================================
IMPORTANT: REUSE EXISTING WORK
==================================================

Before running:

- inspect existing results/
- inspect checkpoints/
- inspect current benchmark scripts
- inspect git status
- reuse already implemented model loaders
- reuse existing K400 checkpoints
- reuse exact SSV2 checkpoints if already downloaded

Do not redo work that already exists.

If a previous PRETRAIN_K400_ON_SSV2 measurement already exists and is valid under the exact same manifest/protocol, verify it before deciding whether a rerun is necessary.

==================================================
FAILURES
==================================================

If a model fails:

DO NOT substitute it.

Record:

FAILED_RUNTIME
FAILED_CHECKPOINT
NO_IMPLEMENTATION
NO_CHECKPOINT

and continue to the next model.

ZeroI2V-B/16 must remain ZeroI2V-B/16.
Do not substitute ZeroI2V-L.

Omnivore-B must remain Omnivore-B.
Do not substitute OmniMAE.

==================================================
FINAL OUTPUT
==================================================

At the end print exactly 11 rows:

Model | Protocol | AccuracyValid | Top1 | Top5 | GFLOPs |
Latency | FPS | Power | Energy/Clip | Total Energy |
VRAM | NS# | Status

Also print:

SSV2 checkpoints used: X/11
K400 pretrained checkpoints used: X/11
Successful device benchmarks: X/11
Valid SSV2 accuracy models: X/11
Failed models: X/11

DO NOT PUSH TO GIT.
DO NOT FINE-TUNE.