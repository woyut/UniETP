# UniETP

This is the official implementation of ["UniETP: Unifying Environments for Generalizable Embodied Task Planning"](https://arxiv.org/abs/2607.18062).

## Code structure


| Path                                | Role                                                                                                                                                                                  |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/run_llm.py`                | Evaluation entry point. It loads a task list, starts the selected simulator, and runs evaluation.                                                                                     |
| `runner.py`                         | Runs one episode: reset the environment, query the agent, execute the action, and check the goal.                                                                                     |
| `general_env.py`                    | Shared environment interface used by the episode loop.                                                                                                                                |
| `THOR/`, `VH/`, `Hab/`, `BEHAVIOR/` | Wrappers for simulators.                                                                                                                                                              |
| `scene_graph/`                      | Unified scene graph representation.                                                                                                                                                   |
| `task/`                             | Logic system and affordance knowledge base.                                                                                                                                           |
| `Agent/`                            | The planner. `prompt/` holds the prompts, `models/` calls a hosted API or a local server, `config/` stores generation settings, and `remote/` runs the planner in a separate process. |

## Benchmark downloading

The benchmark files are provided at [here](https://huggingface.co/datasets/woyut/UniETP).

## Evaluation environments

To avoid conflicts among packages, we recommend using separate environments for different simulators.


| Environment    | Used for                                     |
| -------------- | -------------------------------------------- |
| `unietp-env`   | Evaluation driver, AI2-THOR, and VirtualHome |
| `unietp-hab`   | Habitat-Sim and Habitat-Lab checkout         |
| `behavior`     | BEHAVIOR-1K                                  |
| `unietp-agent` | Planner process and model API clients        |


Please follow the following steps to build the environments as needed.

### AI2-THOR

```bash
conda create -n unietp-env python=3.10 pip -y
conda activate unietp-env
python -m pip install -r requirements.txt
```

We will also need the assets from ProcTHOR-10k.

```bash
export UNIETP_ASSETS=/absolute/path/to/unietp-assets
mkdir -p "$UNIETP_ASSETS"
git clone https://github.com/allenai/procthor-10k.git \
  "$UNIETP_ASSETS/procthor-10k"
git -C "$UNIETP_ASSETS/procthor-10k" lfs pull
export PROCTHOR_DATASET_DIR="$UNIETP_ASSETS/procthor-10k"
```



### VirtualHome

Please build `unietp-env` as in AI2-THOR, then install the simulator package and the Linux executable. Keep the data directory extracted next to the executable. On a machine without a display, `xvfb-run` must be on `PATH`. The launcher uses it automatically.

```bash
conda activate unietp-env
python -m pip install --no-deps virtualhome==2.3.0
python -m pip install ipython ipdb

export VH_ROOT=/absolute/path/to/virtualhome-v2.3.0
mkdir -p "$VH_ROOT"
curl -L \
  http://virtual-home.org/release/simulator/v2.0/v2.3.0/linux_exec.zip \
  -o /tmp/virtualhome-linux-v2.3.0.zip
unzip /tmp/virtualhome-linux-v2.3.0.zip -d "$VH_ROOT"
chmod +x "$VH_ROOT/linux_exec.v2.3.0.x86_64"
export VIRTUALHOME_EXECUTABLE="$VH_ROOT/linux_exec.v2.3.0.x86_64"
```



### Habitat

```bash
conda create -n unietp-hab python=3.9.2 cmake=3.14.0 pip -y
conda activate unietp-hab

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124
conda install -y habitat-sim=0.3.3 withbullet headless \
  -c conda-forge -c aihabitat

mkdir -p third_party
git clone https://github.com/facebookresearch/habitat-lab.git \
  third_party/habitat-lab
git -C third_party/habitat-lab checkout \
  a9c8df586d649972e55500a0fbaae1952b1c3483
python -m pip install -e third_party/habitat-lab/habitat-lab
conda install -c conda-forge -y libegl libopengl
python -m pip install pandas Pillow matplotlib tqdm
```

We also need the assets from HSSD and OVMM.

```bash
export HABITAT_DATA_ROOT=/absolute/path/to/habitat-data
mkdir -p "$HABITAT_DATA_ROOT/versioned_data" "$HABITAT_DATA_ROOT/objects"

git clone https://huggingface.co/datasets/hssd/hssd-hab \
  "$HABITAT_DATA_ROOT/versioned_data/hssd-hab"
git -C "$HABITAT_DATA_ROOT/versioned_data/hssd-hab" lfs pull

git clone https://huggingface.co/datasets/ai-habitat/hab_fetch \
  "$HABITAT_DATA_ROOT/versioned_data/hab_fetch"
git -C "$HABITAT_DATA_ROOT/versioned_data/hab_fetch" lfs pull

git clone https://huggingface.co/datasets/ai-habitat/OVMM_objects \
  "$HABITAT_DATA_ROOT/objects/objects_ovmm"
git -C "$HABITAT_DATA_ROOT/objects/objects_ovmm" lfs pull

export HABITAT_DATA_DIR="$HABITAT_DATA_ROOT/versioned_data/hssd-hab"
cp Hab/hssd-hab-articulated++.scene_dataset_config.json \
  "$HABITAT_DATA_DIR/hssd-hab-articulated++.scene_dataset_config.json"
```

The downloaded asset structure should be like:

```text
$HABITAT_DATA_ROOT/
|-- objects/objects_ovmm/train_val/
|   |-- ai2thorhab/configs/objects/
|   |-- amazon_berkeley/configs/
|   |-- google_scanned/{configs,assets}/
|   `-- hssd/configs/objects/
`-- versioned_data/
    |-- hab_fetch/
    |   |-- robots/hab_fetch.urdf
    |   `-- meshes/
    `-- hssd-hab/                                  # HABITAT_DATA_DIR
        |-- hssd-hab-articulated++.scene_dataset_config.json
        |-- scenes-articulated/
        |-- scene_filter_files/articulated_scene_filter_files/
        |-- objects/
        |-- stages/
        |-- urdf/
        |-- semantics/hssd-hab_semantic_lexicon.json
        `-- metadata/
            |-- object_categories_filtered.csv
            |-- fpmodels-with-decomposed.csv
            |-- room_objects.json
            `-- affordance_objects.csv
```



### BEHAVIOR

```bash
set -e
BEHAVIOR_ROOT=third_party/BEHAVIOR-1K
BEHAVIOR_PATCH_ROOT=patches/behavior-1k/v3.7.2
BEHAVIOR_COMMIT=88454bd04f75dc57c00ab1f1a00bcde1ff505950
UNIETP_ROOT="$(pwd)"

mkdir -p third_party
git clone --branch v3.7.2 \
  https://github.com/StanfordVL/BEHAVIOR-1K.git \
  "$BEHAVIOR_ROOT"
git -C "$BEHAVIOR_ROOT" checkout "$BEHAVIOR_COMMIT"
test "$(git -C "$BEHAVIOR_ROOT" rev-parse HEAD)" = "$BEHAVIOR_COMMIT"

git -C "$BEHAVIOR_ROOT" apply --check \
  "$UNIETP_ROOT/$BEHAVIOR_PATCH_ROOT/unietp-behavior-v3.7.2.patch"
git -C "$BEHAVIOR_ROOT" apply \
  "$UNIETP_ROOT/$BEHAVIOR_PATCH_ROOT/unietp-behavior-v3.7.2.patch"

cd "$BEHAVIOR_ROOT"
bash setup.sh --new-env --omnigibson --bddl --dataset

conda activate behavior
python -m pip install aenum pyquaternion Pillow matplotlib tqdm
```

BEHAVIOR-1K assets are licensed separately from this code. Their use is governed by the license presented during `setup.sh --dataset`.

Based on the offical scene files, we need to further build the initial scenes for our benchmark:

```bash
cd /absolute/path/to/UniETP/code
export UNIETP_BENCHMARK_ROOT=/absolute/path/to/downloaded/benchmark
python scripts/materialize_behavior_scenes.py \
  --benchmark "$UNIETP_BENCHMARK_ROOT/BEHAVIOR.json" \
  --behavior-assets-root third_party/BEHAVIOR-1K/datasets/behavior-1k-assets
```

## Model serving

The evaluation process does not load model weights. For closed-source models, we only need to set the environment variables before running evaluations. 


| Provider  | Required            | Optional             |
| --------- | ------------------- | -------------------- |
| OpenAI    | `OPENAI_API_KEY`    | `OPENAI_BASE_URL`    |
| Google    | `GOOGLE_API_KEY`    | `GOOGLE_BASE_URL`    |
| Anthropic | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |


For open-source models, please deploy the model as a separate inference service (we use vLLM in our 
experiments) and set the host and port in `Agent/config/vllm.json`. 

We recommend building a separate environment, `unietp-agent`, for the communication between the agent model and the simulator. 

```bash
conda create -n unietp-agent python=3.10 pip -y
conda activate unietp-agent
python -m pip install \
  numpy Pillow matplotlib 'pydantic>=2' json-repair \
  openai anthropic google-genai
export UNIETP_AGENT_PYTHON="$CONDA_PREFIX/bin/python"
```



## Running evaluation

Evaluation can be launched by running `scripts/run_llm.py` in the simulator environments. 

For example, to evaluate the AI2-THOR tasks:

```bash
conda activate unietp-env
export UNIETP_BENCHMARK_ROOT=/absolute/path/to/benchmark
python -u scripts/run_llm.py \
  --dataset_file "$UNIETP_BENCHMARK_ROOT/THOR.json" \
  --simulator_name THOR \
  --procthor_dataset_dir "$PROCTHOR_DATASET_DIR" \
  --model_name gpt-5.6-sol \
  --model_config_file Agent/config/api.json \
  --nav_action_mode 3 \
  --manip_action_mode 3 \
  --agent_mode remote \
  --agent_auto_start_service \
  --agent_service_python "$UNIETP_AGENT_PYTHON" \
  --agent_save_dir ./Log/examples
```

VirtualHome uses the same environment, `--dataset_file "$UNIETP_BENCHMARK_ROOT/VH.json"`, and `--vh_exec_file "$VIRTUALHOME_EXECUTABLE"`. Habitat uses `unietp-hab`, `--dataset_file "$UNIETP_BENCHMARK_ROOT/Hab.json"`, and `--habitat_data_dir "$HABITAT_DATA_DIR"`. BEHAVIOR uses the `behavior` environment and `--dataset_file "$UNIETP_BENCHMARK_ROOT/BEHAVIOR.json"`.
The command above runs evaluation mode M1. For M2, use `--nav_action_mode 2 --manip_action_mode 2`. For M3, use `--nav_action_mode 1 --manip_action_mode 1 --max_steps 150 --max_failed_steps 30`.

Outputs are written to `<agent_save_dir>/<dataset-stem>/<task-index>/<scene-index>`.

## Acknowledgements

UniETP is built on four embodied simulation platforms, and we sincerely thank their authors and maintainers for making them publicly available: [AI2-THOR](https://github.com/allenai/ai2thor) together with the [ProcTHOR-10k](https://github.com/allenai/procthor-10k) houses, [VirtualHome](https://github.com/xavierpuigf/virtualhome), [Habitat](https://github.com/facebookresearch/habitat-sim) ([habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab)), and [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K).

Parts of our code are adapted from the following open-source projects, and we are grateful to their authors:

- [partnr-planner](https://github.com/facebookresearch/partnr-planner) (MIT License)
- [habitat-lab](https://github.com/facebookresearch/habitat-lab) (MIT License)
- [OmniGibson](https://github.com/StanfordVL/BEHAVIOR-1K) (MIT License)
- [VisualAgentBench](https://github.com/THUDM/VisualAgentBench) (Apache License 2.0)
- [VirtualHome](https://github.com/xavierpuigf/virtualhome) (MIT License)

We also thank the creators of the [Habitat Synthetic Scenes Dataset (HSSD)](https://huggingface.co/datasets/hssd/hssd-hab), the [OVMM objects](https://huggingface.co/datasets/ai-habitat/OVMM_objects) and the [BEHAVIOR-1K assets](https://github.com/StanfordVL/BEHAVIOR-1K). Third-party code and data remain subject to their original licenses.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for file-level provenance and the corresponding license texts.
