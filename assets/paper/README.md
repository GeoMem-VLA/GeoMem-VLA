# Manuscript figures and reported results

These assets were selected from the author-supplied `3D_Memory_VLA.zip` manuscript package. PDF figures are preserved unchanged; their PNG counterparts are direct renders for GitHub display. Execution-sequence PNGs are copied unchanged. The archive and asset SHA-256 hashes are recorded in [provenance.json](provenance.json).

## Figure sources

| Release asset | Manuscript source | Context |
| --- | --- | --- |
| [Architecture](architecture.pdf) | `figures/GeoMemVLA_v4.pdf`; `figures/architecture.tex` | Paper architecture; the simplified release configuration does not enable every research variant shown. |
| [Real-world tasks](real_world_tasks.pdf) | `figures/task.pdf`; `sections/04_real_robot.tex` | Five tasks, 60 demonstrations per task, 10 evaluation rollouts per task. |
| [Geometry and memory](geometry_memory_case.pdf) | `figures/cases/case_figure.pdf`; `sections/04_case_studies.tex`; `sections/08_case_details.tex` | Offline RoboTwin demonstration replay using a 200k-step checkpoint and capacity-eight memory banks. |
| [Astribot successes](astribot_success.png) | `figures/real_world_cases/astribot_success.png` | Selected successful execution sequences. |
| [AgileX successes](agilex_success.png) | `figures/real_world_cases/agilex_success.png` | Selected successful execution sequences; times are after editing. |
| [Astribot failures](astribot_failure.png) | `figures/real_world_cases/astribot_failure.png` | Selected unsuccessful execution sequences. |
| [AgileX failures](agilex_failure.png) | `figures/real_world_cases/agilex_failure.png` | Selected unsuccessful execution sequences. |

The archive also contains an older architecture PNG and two different case-figure PDFs. The selected figures above follow the actual `includegraphics` paths in the manuscript's architecture and case-study sections.

## Reported metrics

[results.json](results.json) transcribes the GeoMem-VLA scores and component ablations used in the root README:

- LIBERO: `tables/tab_libero_main.tex` (50 trials per task, four 10-task suites).
- RoboTwin: `tables/tab_robotwin_main.tex` (100 episodes per task and setting, 50 tasks).
- Component ablations: `tables/tab_ablation.tex`.
- Real-world tasks: `figures/task.pdf` and `sections/04_real_robot.tex`.
- Evaluation protocols: `sections/04_experiments.tex` and `sections/09_real_world_exper_details.tex`.

These values were checked against the supplied manuscript, not recomputed from raw rollout logs. Baseline bars in the real-world figure are reproduced as part of the original paper figure. No new leaderboard or performance measurements are claimed.

## Relationship to the public configuration

The README's 20,000-step quick start uses group size and memory length 16, a LIBERO spatial subset, and a one-task evaluation. It is not the paper's benchmark reproduction recipe. The case figure instead processes 16 sampled frames through capacity-eight banks, and its EEF reconstruction examples use a research configuration with EEF targets; the basic RoboTwin recipe disables EEF supervision. Real-robot deployment code and the research checkpoints are not bundled with these illustrations.
