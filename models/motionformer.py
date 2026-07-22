# import torch.nn as nn
# import sys
# from pathlib import Path

# class MotionFormer(nn.Module):
#     def __init__(self, num_classes=400):
#         super().__init__()
#         repo_path = Path(__file__).parent / 'motionformer_repo'
#         if repo_path.exists():
#             sys.path.insert(0, str(repo_path))
#             try:
#                 # e.g., from motionformer import VideoTransformer
#                 from motionformer import VideoTransformer
#                 self.model = VideoTransformer(num_classes=num_classes)
#                 return
#             except ImportError:
#                 pass
#         raise NotImplementedError(
#             "MotionFormer requires the official Facebook Research repository.\n"
#             "Clone https://github.com/facebookresearch/Motionformer and place it as './models/motionformer_repo'.\n"
#             "Then adapt the import (e.g., from motionformer import VideoTransformer)."
#         )
#     def forward(self, x):
#         return self.model(x)

import sys
from pathlib import Path
import torch.nn as nn

class MotionFormer(nn.Module):
    def __init__(self, num_classes=400):
        super().__init__()
        repo = Path(__file__).parent.parent / 'third_party' / 'others' / 'motionformer'
        if repo.exists():
            sys.path.insert(0, str(repo))
            try:
                from motionformer import MotionFormer as _MF
                self.model = _MF(num_classes=num_classes)
                return
            except ImportError:
                pass
        raise NotImplementedError(
            "MotionFormer not found. Place the official code in third_party/others/motionformer/"
        )
    def forward(self, x):
        return self.model(x)