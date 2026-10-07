# BEHAVIOR-1K v3.7.2 compatibility patch

`unietp-behavior-v3.7.2.patch` applies the UniETP compatibility changes to the
following upstream revision:

- Repository: `https://github.com/StanfordVL/BEHAVIOR-1K.git`
- Tag: `v3.7.2`
- Commit: `88454bd04f75dc57c00ab1f1a00bcde1ff505950`

The patch must not be applied to another BEHAVIOR-1K revision. It:

1. construct segmentation-map indices with `torch.stack` and an explicit scalar
   random index;
2. force `BaseRobot` to load with `visual_only=True`;
3. force vision sensors to use a 512 x 512 image size.

From the UniETP repository root, apply it with:

```bash
BEHAVIOR_ROOT=third_party/BEHAVIOR-1K
PATCH_ROOT=patches/behavior-1k/v3.7.2
UNIETP_ROOT="$(pwd)"

test "$(git -C "$BEHAVIOR_ROOT" rev-parse HEAD)" = \
  "88454bd04f75dc57c00ab1f1a00bcde1ff505950"
git -C "$BEHAVIOR_ROOT" apply --check \
  "$UNIETP_ROOT/$PATCH_ROOT/unietp-behavior-v3.7.2.patch"
git -C "$BEHAVIOR_ROOT" apply \
  "$UNIETP_ROOT/$PATCH_ROOT/unietp-behavior-v3.7.2.patch"
```

The main repository README contains the complete clone, revision-check,
installation, and verification procedure. Restart existing Python and
OmniGibson processes after copying these files.

The patch is derived from OmniGibson and is distributed under the upstream
license included as `LICENSE` in this directory.
