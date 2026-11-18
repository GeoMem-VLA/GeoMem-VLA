# RoboTwin secondary example

The primary release walkthrough is LIBERO; this adapter is retained for
RoboTwin 2.0. Install [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)
separately and prepare its LeRobot-format data with `train_files/modality.json`
in each dataset's `meta` directory.

Export `BASE_VLM`, `VGGT_CKPT`, `VGGT_REPO`, and `ROBOTWIN_DATA_ROOT` as described
in the root `.env.example`, activate the model environment, then run:

```bash
bash examples/Robotwin/train_files/run_robotwin_train.sh
```

Defaults: batch/group/memory 16/16/16, gradient accumulation 8, 20,000 training steps,
state conditioning and q99 normalization. EEF supervision is disabled because
the official joint-space data does not provide `observation.eef`. The optional
EEF conversion utilities remain available for separate experiments.

## Optional: EEF track in the VGGT point-cloud frame

A research configuration, separate from the basic recipe, supervises the GeoFlow
head with the current VGGT point map and the future dual-arm EEF track in the same
normalized frame (`QwenGeoGR00T_RoboTwin`). For each sample, `_eef_world_to_vggt`
maps the world-frame EEF positions into the head camera with the per-frame
extrinsic and projects the first four (current) positions onto the image with the
intrinsic. The metric-to-VGGT scale is the median of the VGGT depth at those
gripper pixels divided by the camera-frame depth. Samples whose gripper is out of
view or behind the camera, or whose scale is more than 4x from the batch median,
get zero EEF weight; `geo/eef_valid_frac` logs the fraction kept. The track and the
point map are then normalized together.

The EEF positions and camera parameters are read from the raw RoboTwin HDF5 files,
so this mode needs the raw zips and a LeRobot dataset converted from those same zips
with `convert_robotwin_to_lerobot.py`. A separately downloaded LeRobot dataset is a
different capture and will not match. Run the two scripts below once per dataset.
Each one checks every episode against the zip and aborts that task on a mismatch,
appends one column in place, leaves the videos untouched, skips datasets that
already have it, and writes atomically:

```bash
python examples/Robotwin/train_files/add_eef_to_lerobot.py \
  --lerobot-root "$ROBOTWIN_DATA_ROOT" --data-root /path/to/RoboTwin/data/dataset
python examples/Robotwin/train_files/add_campose_to_lerobot.py \
  --lerobot-root "$ROBOTWIN_DATA_ROOT" --data-root /path/to/RoboTwin/data/dataset
```

Do not substitute `add_eef_via_fk.py` here: its positions are in the URDF root
frame, while this mode applies the world-to-camera extrinsic.

Then train with the same environment variables as above:

```bash
bash examples/Robotwin/train_files/run_robotwin_train_v2.sh
```

This uses `train_files/starvla_robotwin_v2.yaml` and the `robotwin_all_50_v2`
mixture, which also reads `observation.eef` and `observation.campose`. Batch, group
and memory length default to 8/8/8, and the other launch options match the basic
recipe. The root `.env.example` sets 16/16/16, so edit or unset `PER_DEVICE_BATCH_SIZE`,
`GROUP_SIZE`, and `MEM_LENGTH` if you source it and want 8/8/8. This configuration
also adds a second-difference action smoothness loss (`action_smooth_coeff`) and
logs the geometry diagnostic probe every 100 steps.

For evaluation, see `eval_files/start_eval.sh --help`. Provide the model and
simulator Python interpreters via `STARVLA_PYTHON` and `ROBOTWIN_PYTHON` when
using that orchestrator. Its external simulator setup is separate from model
installation. The server defaults to streaming memory; clients reset it at
episode boundaries. These defaults are not claimed to reproduce published scores.
