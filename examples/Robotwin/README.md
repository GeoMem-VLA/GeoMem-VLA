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

For evaluation, see `eval_files/start_eval.sh --help`. Provide the model and
simulator Python interpreters via `STARVLA_PYTHON` and `ROBOTWIN_PYTHON` when
using that orchestrator. Its external simulator setup is separate from model
installation. The server defaults to streaming memory; clients reset it at
episode boundaries. These defaults are not claimed to reproduce published scores.
