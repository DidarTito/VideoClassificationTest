# SSV2 head-transfer / linear-probe remaining 11

For models without an exact released SSV2 checkpoint, we retained the pretrained Kinetics-400/video backbone, froze all pretrained backbone parameters, extracted fixed video representations from the locally available SSV2 training subset, and trained only a newly initialized 174-class linear classification head. Therefore these measurements represent SSV2 linear-probe/head-transfer accuracy rather than full-network fine-tuning.

86,680 official training videos are currently present locally rather than the complete SSV2 train split (168,913). Head training used up to 50 present train videos per class and up to 10 present validation videos per class, seed 20260923, disjoint from the fixed 1000-clip evaluation manifest.

Completed 11/11.

| Model | Top-1 | Top-5 | Val Top-1 | D | Train N | LR | Backbone unchanged | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| videoswin-t | 10.6 | 27.7 | 12.2036 | 768 | 8695 | 0.01 | true | COMPLETE_HEAD_TRAINING |
| videoswin-s | 11.1 | 28.6 | 11.0468 | 768 | 8695 | 0.003 | true | COMPLETE_HEAD_TRAINING |
| video-focalnet-t | 9.5 | 27.3 | 10.989 | 768 | 8695 | 0.001 | true | COMPLETE_HEAD_TRAINING |
| video-focalnet-s | 11.1 | 27.1 | 11.7409 | 768 | 8695 | 0.001 | true | COMPLETE_HEAD_TRAINING |
| dualformer-t | 11.2 | 29.6 | 11.0468 | 512 | 8695 | 0.01 | true | COMPLETE_HEAD_TRAINING |
| dualformer-s | 13.3 | 30.5 | 11.9144 | 768 | 8695 | 0.01 | true | COMPLETE_HEAD_TRAINING |
| dualformer-b-in21k | 12.1 | 31.2 | 12.4928 | 1024 | 8695 | 0.01 | true | COMPLETE_HEAD_TRAINING |
| mvit-v1-b-16x4 | 9.2 | 26.0 | 10.2371 | 768 | 8695 | 0.01 | true | COMPLETE_HEAD_TRAINING |
| mvit-v1-b-32x3 | 10.5 | 29.2 | 11.6252 | 768 | 8695 | 0.01 | true | COMPLETE_HEAD_TRAINING |
| omnivore-b-in21k | 14.7 | 33.6 | 14.6327 | 1024 | 8695 | 0.003 | true | COMPLETE_HEAD_TRAINING |
| zeroi2v-b16-8f | 13.1 | 32.9 | 14.2279 | 768 | 8695 | 0.003 | true | COMPLETE_HEAD_TRAINING |
