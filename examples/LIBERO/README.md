# LIBERO basic workflow

Follow the repository README for installation, external paths, training and
one-task evaluation. The public default trains on `libero_spatial_state` and
evaluates the first spatial task once. It is not a tuned benchmark recipe.

## Data

Download the LeRobot LIBERO datasets, for example
[IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot).
Set `LIBERO_DATA_ROOT` to the parent directory containing the dataset folders.
Each dataset needs `train_files/modality.json` copied to `meta/modality.json`.
The optional `data_preparation.sh` downloads spatial by default into
`LIBERO_DATA_ROOT` (or `DEST`). Set `LIBERO_SUITES="spatial object goal 10"`
to download all four.
To train all four, provide all datasets and set `DATA_MIX=libero_all_state`.

## Simulator environment

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) separately.
In its Python environment, install the evaluation dependencies:

```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4 mujoco==3.2.3
```

Activate the model environment for `eval_files/run_policy_server.sh`, and the
simulator environment for `eval_files/eval_libero.sh`. Both require `CKPT`;
the client also requires `LIBERO_HOME`. Export values from `.env` in each shell.
Optional interpreter overrides: `STARVLA_PYTHON` and `LIBERO_PYTHON`.
The default state input matches the supplied state-conditioned training recipe;
set `LIBERO_SEND_STATE=0` only for a checkpoint trained without state.
