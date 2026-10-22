import dataclasses
import hashlib
import json
import logging
import math
import os
import pathlib
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

# Per-suite episode length caps (longest training demo + margin). Keyed by the
# suite name, which also matches each task's `problem_folder` — so combined
# suites like `libero_all` can look up the right cap per task.
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}

# Diagnostic-only override of the caps above: LIBERO_MAX_STEPS=<int> forces one cap for every
# suite. Raising it separates "the policy cannot do the task" from "the policy ran out of budget",
# but the resulting success rates are NOT comparable to the standard caps and must never be
# reported as LIBERO numbers.
_MAX_STEPS_OVERRIDE = os.getenv("LIBERO_MAX_STEPS", "").strip()

# LIBERO_SAVE_ACTIONS=1 dumps the executed action sequence of every episode next to its mp4 as
# a [T, action_dim] .npy. Needed to measure action smoothness (e.g. how much the arm jitters at
# chunk boundaries under few-step flow-matching integration) instead of inferring it from
# success rates, which are far noisier.
_SAVE_ACTIONS = os.getenv("LIBERO_SAVE_ACTIONS", "0").strip() not in ("", "0", "false", "False")
_SAVE_VIDEOS = os.getenv("LIBERO_SAVE_VIDEOS", "1").strip() not in ("", "0", "false", "False")


def _resolve_max_steps(problem_folder: str) -> int:
    if _MAX_STEPS_OVERRIDE:
        return int(_MAX_STEPS_OVERRIDE)
    return MAX_STEPS_BY_SUITE[problem_folder]

# The four standard suites that `libero_all` expands to (matches the training
# `libero_all` data mixture). Evaluated back-to-back within a single run.
LIBERO_ALL_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = -1  # If > 0, limit the number of tasks evaluated (smoke / quick check). -1 = run all.
    task_ids: str = ""  # Optional comma-separated, zero-based task IDs in one suite.
    episode_start: int = 0  # Preserve original init-state IDs when evaluating a held-out slice.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    post_process_action: bool = True

    job_name: str = "test"


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    if args.num_trials_per_task <= 0 or args.episode_start < 0:
        raise ValueError("num_trials_per_task must be positive and episode_start nonnegative")

    # Set random seed
    np.random.seed(args.seed)

    if _MAX_STEPS_OVERRIDE:
        logging.warning(
            f"*** LIBERO_MAX_STEPS={_MAX_STEPS_OVERRIDE} overrides the per-suite episode caps "
            f"{MAX_STEPS_BY_SUITE} — DIAGNOSTIC ONLY, these success rates are not comparable to "
            f"standard LIBERO results ***"
        )

    # Initialize LIBERO task suite(s). "libero_all" expands to the four standard
    # suites; any other name resolves to that single suite.
    if args.task_suite_name == "libero_all":
        suite_names = LIBERO_ALL_SUITES
    elif args.task_suite_name in MAX_STEPS_BY_SUITE:
        suite_names = [args.task_suite_name]
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    benchmark_dict = benchmark.get_benchmark_dict()

    # Flatten into a list of (task_suite, task_id) so the eval loop iterates
    # uniformly whether we run one suite or several.
    task_entries = []
    for suite_name in suite_names:
        suite = benchmark_dict[suite_name]()
        for task_id in range(suite.n_tasks):
            task_entries.append((suite, task_id))
    num_tasks_in_suite = len(task_entries)
    if args.task_ids:
        if len(suite_names) != 1:
            raise ValueError("task_ids requires a single suite, not libero_all")
        selected_ids = [int(value.strip()) for value in args.task_ids.split(",")]
        if len(set(selected_ids)) != len(selected_ids) or any(
            value < 0 or value >= num_tasks_in_suite for value in selected_ids
        ):
            raise ValueError(f"task_ids must be distinct IDs in [0, {num_tasks_in_suite})")
        task_entries = [task_entries[value] for value in selected_ids]
        num_tasks_in_suite = len(task_entries)
    logging.info(
        f"Task suite: {args.task_suite_name} "
        f"({num_tasks_in_suite} tasks across {len(suite_names)} suite(s): {suite_names})"
    )

    # args.video_out_path = f"{date_base}+{args.job_name}"

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client_class = ModelClient
    if os.getenv("LIBERO_ROUTER_CONFIG"):
        from examples.LIBERO.eval_files.model2libero_routed import RoutedModelClient
        client_class = RoutedModelClient
    client_model = client_class(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
    )
    # The client path used to be a log label only: an already-running server
    # could silently evaluate a different checkpoint (seen in historical runs).
    server_meta = client_model._server_metadata
    if args.pretrained_path:
        expected = pathlib.Path(args.pretrained_path).resolve()
        actual_path = server_meta.get("ckpt_path")
        if not actual_path or pathlib.Path(actual_path).resolve() != expected:
            raise ValueError(
                f"Checkpoint mismatch: requested {expected}, server loaded {actual_path!r}. "
                "Restart the policy server with the requested checkpoint."
            )
    (pathlib.Path(args.video_out_path) / "server_metadata.json").write_text(
        json.dumps(server_meta, indent=2) + "\n"
    )

    # Optional smoke-test cap (still useful for quick verification with -1 = full run).
    n_eval_tasks = num_tasks_in_suite if args.max_tasks <= 0 else min(args.max_tasks, num_tasks_in_suite)
    logging.info(f"Evaluating {n_eval_tasks} of {num_tasks_in_suite} tasks (max_tasks={args.max_tasks})")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for eval_idx in tqdm.tqdm(range(n_eval_tasks)):
        # Resolve the suite and per-suite task id for this eval slot.
        task_suite, task_id = task_entries[eval_idx]
        # Get task
        task = task_suite.get_task(task_id)

        # Episode cap per task: for combined suites (e.g. libero_all) each task
        # may belong to a different sub-suite, so key off its problem_folder.
        max_steps = _resolve_max_steps(task.problem_folder)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        episode_stop = args.episode_start + args.num_trials_per_task
        if episode_stop > len(initial_states):
            raise ValueError(f"Requested init states [{args.episode_start}, {episode_stop}), available {len(initial_states)}")

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.episode_start, episode_stop)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            # episode_index identifies the scene (paired with task_description), so the server's
            # sampling seed is tied to the scene rather than to the run's episode counter.
            client_model.reset(task_description=task_description, episode_index=episode_idx)
            episode_env_seed = args.seed
            if os.getenv("LIBERO_PAIR_EPISODE_SEEDS", "0") == "1":
                episode_env_seed = (args.seed * 1_000_003 + int(client_model.episode_id)) % (2**31 - 1)
                np.random.seed(episode_env_seed)
                env.seed(episode_env_seed)
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            full_actions = []
            first_observation_hash = None

            logging.info(f"Starting episode {episode_idx + 1}...")
            step = 0

            # full_actions = np.load("./debug/action.npy")

            while t < max_steps + args.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # Save preprocessed image for replay video
                if _SAVE_VIDEOS:
                    replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = {  #
                    "observation.primary": np.expand_dims(img, axis=0),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # align key with model API --> two images provided here --> check training
                # NOTE: `state` is always available to the client. It forwards the full state only
                # with LIBERO_SEND_STATE=1; event-guided checkpoints may also derive and send their
                # causal 9-D event_aux from it, as advertised by server metadata.
                example_dict = {
                    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "lang": observation["instruction"][0],
                    "state": observation["observation.state"][0],
                }
                if first_observation_hash is None and os.getenv("LIBERO_RESULTS_JSONL", ""):
                    digest = hashlib.sha256()
                    for value in (*example_dict["image"], example_dict["state"]):
                        digest.update(np.ascontiguousarray(value).tobytes())
                    first_observation_hash = digest.hexdigest()

                start_time = time.time()

                response = client_model.step(example=example_dict, step=step)

                end_time = time.time()
                # print(f"time: {end_time - start_time}")

                # #
                raw_action = response["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(
                        f"Unexpected action sizes: "
                        f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                        f"Falling back to LIBERO_DUMMY_ACTION."
                    )
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)

                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            if _SAVE_VIDEOS:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            full_actions = np.stack(full_actions)
            if _SAVE_ACTIONS:
                np.save(
                    pathlib.Path(args.video_out_path)
                    / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy",
                    full_actions,
                )

            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            # Optional per-episode records for paired inference interventions.
            results_jsonl = os.getenv("LIBERO_RESULTS_JSONL", "")
            if results_jsonl:
                with open(results_jsonl, "a", encoding="utf-8") as result_file:
                    result_file.write(json.dumps({
                        "suite": task.problem_folder, "task_id": int(task_id),
                        "task": task_description, "episode_index": int(episode_idx),
                        "episode_id": int(client_model.episode_id), "success": bool(done),
                        "executed_steps": len(full_actions), "env_seed": args.seed,
                        "episode_env_seed": episode_env_seed,
                        "first_observation_sha256": first_observation_hash,
                        "condition": server_meta.get("geo_history_intervention", "none"),
                        "checkpoint": server_meta.get("ckpt_path"),
                    }) + "\n")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
