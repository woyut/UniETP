# Third-Party Notices

This repository contains code and patches adapted from the third-party projects
listed below. Those portions remain subject to their respective upstream
licenses. The corresponding license texts are included in the `LICENSES/`
directory.

The files in `LICENSES/` apply only to the third-party material identified in
this document; they do not establish a license for other repository contents.

## VisualAgentBench

- Project: VisualAgentBench
- Upstream repository: https://github.com/THUDM/VisualAgentBench
- Source revision: commit
  `9055fc299c366ef34700d1710215fb60a0d8c35e`
- License: Apache License 2.0
- License copy: `LICENSES/VisualAgentBench-Apache-2.0.txt`
- UniETP locations:
  - `BEHAVIOR/env_utils.py`

The adapted code has been modified for integration with UniETP and the version
of OmniGibson used by this project.

## BEHAVIOR-1K / OmniGibson

- Project: BEHAVIOR-1K, including OmniGibson
- Upstream repository: https://github.com/StanfordVL/BEHAVIOR-1K
- Patch base: tag `v3.7.2`, commit
  `88454bd04f75dc57c00ab1f1a00bcde1ff505950`
- Copyright: Copyright (c) 2023 Stanford Vision and Learning Group
- License: MIT License
- License copy: `LICENSES/BEHAVIOR-1K-MIT.txt`
- UniETP locations:
  - `BEHAVIOR/ModifiedSemanticActionPrimitive.py`
  - `patches/behavior-1k/v3.7.2/unietp-behavior-v3.7.2.patch`

The Python implementation is adapted from OmniGibson semantic action primitive
and particle-system implementations. The patch contains UniETP compatibility
changes for the pinned upstream revision.

## PARTNR

- Project: partnr-planner
- Upstream repository: https://github.com/facebookresearch/partnr-planner
- Source revision: commit
  `ddfff19f4b6c098a31edea4d19e7b75db72433c2`
- Copyright: Copyright (c) Meta Platforms, Inc. and its affiliates
- License: MIT License
- License copy: `LICENSES/PARTNR-MIT.txt`
- UniETP locations:
  - `Hab/hab_SG_utils.py`
  - `Hab/hab_action_utils.py`
  - `scene_graph/graph_schema/graph.py`

The adapted code has been modified for UniETP's unified scene graph and Habitat
environment integration.

## Habitat-Lab

- Project: Habitat-Lab
- Upstream repository: https://github.com/facebookresearch/habitat-lab
- Required upstream revision:
  `a9c8df586d649972e55500a0fbaae1952b1c3483`
- Copyright: Copyright (c) Meta Platforms, Inc. and its affiliates
- License: MIT License
- License copy: `LICENSES/Habitat-Lab-MIT.txt`
- UniETP location:
  - `Hab/hab_action_utils.py` (`RearrangeSim.safe_snap_point`-derived logic)

The adapted code has been modified for UniETP's Habitat action implementation.

## VirtualHome

- Project: VirtualHome
- Upstream repository: https://github.com/xavierpuigf/virtualhome
- Source revision: commit
  `58970fd80951c2eaa1af713e0917d1a105353ad8`
- Copyright: Copyright (c) 2022 MIT
- License: MIT License
- License copy: `LICENSES/VirtualHome-MIT.txt`
- UniETP location:
  - `VH/unity_launcher_patch.py`

The adapted `UnityLauncher` implementation adds headless execution, configurable
logging, and process-group cleanup. The upstream VirtualHome source attributes
parts of this component to Unity ML-Agents and the AI2-THOR controller, both
licensed under Apache License 2.0.

