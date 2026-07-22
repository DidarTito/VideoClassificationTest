# Third-party source pinning plan

The model adapters import upstream research code from `third_party/`. Each
current directory is an independent Git clone. Adding these directories with
plain `git add` would create embedded gitlinks without a `.gitmodules` file,
which produces a broken clone for downstream users.

After the parent repository is initialized and its remote history is checked,
replace the local clones with formal submodules pinned to these audited
commits:

| Path | Upstream | Commit |
|---|---|---|
| `third_party/ActionCLIP` | `https://github.com/sallymmx/ActionCLIP.git` | `31c34df17dce` |
| `third_party/DualFormer` | `https://github.com/sail-sg/dualformer.git` | `058c10c6a078` |
| `third_party/omnivore` | `https://github.com/facebookresearch/omnivore.git` | `1d55abdc8dfc` |
| `third_party/SlowFast` | `https://github.com/facebookresearch/SlowFast.git` | `287ec0076846` |
| `third_party/TimeSformer` | `https://github.com/facebookresearch/TimeSformer.git` | `a5ef29a7b726` |
| `third_party/UniFormer` | `https://github.com/Sense-X/UniFormer.git` | `52a0415e4e0d` |
| `third_party/Video-FocalNets` | `https://github.com/TalalWasim/Video-FocalNets.git` | `39eba998ebaf` |
| `third_party/Video-Swin-Transformer` | `https://github.com/SwinTransformer/Video-Swin-Transformer.git` | `db018fb88962` |
| `third_party/VideoMAE` | `https://github.com/MCG-NJU/VideoMAE.git` | `14ef8d856287` |

Before conversion, confirm that no source edits appeared after this audit.
Only generated `__pycache__` directories were untracked in the audited
SlowFast and VideoMAE clones; the upstream source trees themselves were clean.

A safe conversion sequence is:

1. Record each current `origin` and full `HEAD` hash.
2. Move the existing directory to a temporary backup outside the repository.
3. Run `git submodule add <upstream> <path>` from the initialized parent.
4. Check out the audited full commit inside the submodule.
5. Compare the submodule working tree with the backup before removing it.
6. Commit `.gitmodules` and the nine gitlinks together.

Do not copy third-party checkpoint files into the submodules. Upstream code and
weights retain their own licenses and citation requirements.

