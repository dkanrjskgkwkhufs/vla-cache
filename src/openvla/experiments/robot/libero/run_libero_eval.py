"""
run_libero_eval.py  (VLA-Cache + SP-VLA Step Skipping)

VLA-Cache(토큰 KV 재사용) 평가 스크립트에 SP-VLA의 action 재사용(step skipping)을 추가한 버전.

  - use_vla_cache : 모델을 돌릴 때 정적 토큰의 KV를 재사용 (VLA-Cache)
  - use_step_skip : 평면 이동 구간에서는 모델을 건너뛰고 ridge regression으로 action 예측 (SP-VLA)

핵심 변경점
  1) ActionScheduler 클래스 추가 (SP-VLA modeling_prismatic.py의 save_actions / skip 조건 / fit_next_action 이식)
  2) prev_img = "직전 env 프레임"이 아니라 "마지막으로 모델을 돌린 프레임"
     → last_caches(KV, attention)가 만들어진 프레임과 비교 기준 프레임을 일치시킴
  3) skip한 스텝에서는 last_caches / last_infer_img 를 갱신하지 않음
  4) skip rate, 추론 스텝 수, 재사용 토큰 수, latency 로깅

Usage:
    # 4가지 조합 비교
    python experiments/robot/libero/run_libero_eval.py --use_vla_cache False --use_step_skip False   # baseline
    python experiments/robot/libero/run_libero_eval.py --use_vla_cache True  --use_step_skip False   # cache only
    python experiments/robot/libero/run_libero_eval.py --use_vla_cache False --use_step_skip True    # skip only
    python experiments/robot/libero/run_libero_eval.py --use_vla_cache True  --use_step_skip True    # cache + skip
"""

import copy
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


# SP-VLA run_libero.py 의 suite별 skip 하이퍼파라미터
SKIP_PARAMS = {
    "libero_spatial": dict(z_xy_rate_skip=0.4, z_max_skip=0.3, step_skip=1),
    "libero_object":  dict(z_xy_rate_skip=0.6, z_max_skip=0.5, step_skip=2),
    "libero_goal":    dict(z_xy_rate_skip=1.2, z_max_skip=0.3, step_skip=2),
    "libero_10":      dict(z_xy_rate_skip=1.0, z_max_skip=0.3, step_skip=2),
}

MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,   # longest training demo has 254 steps
    "libero_goal": 300,     # longest training demo has 270 steps
    "libero_10": 520,       # longest training demo has 505 steps
    "libero_90": 400,       # longest training demo has 373 steps
}


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Acceleration switches
    #################################################################################################################
    use_vla_cache: bool = True                       # VLA-Cache: 정적 토큰 KV 재사용
    use_step_skip: bool = True                       # SP-VLA: action 재사용 (step skipping)

    # Step skipping 하이퍼파라미터 (None이면 SKIP_PARAMS의 suite 기본값 사용)
    z_xy_rate_skip: Optional[float] = None           # max|z| / max(|x|,|y|) < 이 값이어야 skip
    z_max_skip: Optional[float] = None               # |z| < 이 값이어야 skip
    step_skip: Optional[int] = None                  # 최근 몇 개 action이 위 조건을 만족해야 하는지
    action_buffer_size: int = 6                      # action 버퍼 크기 (SP-VLA: 6)
    min_generate_ratio: float = 4 / 6                # 버퍼 내 VLA 생성 action 비율 하한 (SP-VLA: 4/6)

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "checkpoints/openvla-7b-finetuned-libero-spatial"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    save_video: bool = True                          # Save rollout video per episode

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


class ActionScheduler:
    """
    SP-VLA step skipping (prismatic/extern/hf/modeling_prismatic.py 의 compressor 로직을 모델 밖으로 이식).

    버퍼에는 normalize/invert_gripper_action 적용 "전"의 action(모델 원출력 공간)을 저장한다.
    """

    def __init__(self, z_xy_rate_skip, z_max_skip, step_skip, buffer_size=6, min_generate_ratio=4 / 6):
        self.z_xy_rate_skip = z_xy_rate_skip
        self.z_max_skip = z_max_skip
        self.step_skip = step_skip
        self.buffer_size = buffer_size
        self.min_generate_ratio = min_generate_ratio
        self.reset()

    def reset(self):
        self.actions = []          # 최근 action (최대 buffer_size개)
        self.types = []            # "generate" | "fit"
        self.num_generate = 0      # 버퍼 안의 VLA 생성 action 수
        self.episode_skips = 0
        self.episode_infers = 0

    # --- SP-VLA save_actions ---------------------------------------------------------------------
    def record(self, action, generated: bool):
        self.actions.append(np.array(action, dtype=np.float64, copy=True))
        self.types.append("generate" if generated else "fit")
        if generated:
            self.num_generate += 1
            self.episode_infers += 1
        else:
            self.episode_skips += 1

        if len(self.actions) > self.buffer_size:
            self.actions.pop(0)
            if self.types.pop(0) == "generate":
                self.num_generate -= 1

    # --- SP-VLA skip 조건 --------------------------------------------------------------------------
    def should_skip(self) -> bool:
        # 조건 1: 버퍼 내 VLA 생성 action 비율 (SP-VLA: save_generate_step / save_action_number >= 4/6)
        if self.num_generate / self.buffer_size < self.min_generate_ratio:
            return False

        # 조건 2: 최근 step_skip개 action이 모두 "평면 이동"
        if len(self.actions) < self.step_skip:
            return False
        for i in range(1, self.step_skip + 1):
            a = self.actions[-i]
            max_xy = max(abs(a[0]), abs(a[1]), 1e-4)
            max_z = abs(a[2])
            if not (max_z / max_xy < self.z_xy_rate_skip and max_z < self.z_max_skip):
                return False
        return True

    # --- SP-VLA fit_next_action (ridge regression) ----------------------------------------------
    def predict(self) -> np.ndarray:
        total = np.array(self.actions)
        Y = total[:, :3]                                         # (N, 3) xyz translation
        X = np.arange(len(Y)).reshape(-1, 1)
        X = np.hstack([X, np.ones_like(X)])                      # (N, 2)

        lambda_reg = 1e-4
        beta = np.linalg.solve(X.T @ X + lambda_reg * np.eye(2), X.T @ Y)   # (2, 3)
        next_xyz = beta[0] * len(Y) + beta[1]
        next_xyz = np.clip(next_xyz, -0.9, 0.9)

        # 증가량이 과거 최대 증가량을 넘지 않도록 제한
        delta_max = np.max(np.abs(np.diff(Y, axis=0)), axis=0)
        last_xyz = Y[-1]
        next_xyz = last_xyz + np.clip(next_xyz - last_xyz, -delta_max, delta_max)

        # 회전 + gripper는 직전 action 그대로 (SP-VLA 코드 동작)
        return np.hstack([next_xyz, total[-1, 3:]])


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Step skipping 하이퍼파라미터 결정
    defaults = SKIP_PARAMS.get(cfg.task_suite_name, SKIP_PARAMS["libero_spatial"])
    if cfg.z_xy_rate_skip is None:
        cfg.z_xy_rate_skip = defaults["z_xy_rate_skip"]
    if cfg.z_max_skip is None:
        cfg.z_max_skip = defaults["z_max_skip"]
    if cfg.step_skip is None:
        cfg.step_skip = defaults["step_skip"]

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    scheduler = ActionScheduler(
        z_xy_rate_skip=cfg.z_xy_rate_skip,
        z_max_skip=cfg.z_max_skip,
        step_skip=cfg.step_skip,
        buffer_size=cfg.action_buffer_size,
        min_generate_ratio=cfg.min_generate_ratio,
    )

    # Initialize local logging
    mode = f"cache{int(cfg.use_vla_cache)}-skip{int(cfg.use_step_skip)}"
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{mode}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    log_file.write(f"Config: {json.dumps({k: str(v) for k, v in vars(cfg).items()}, ensure_ascii=False)}\n")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)
    max_steps = MAX_STEPS[cfg.task_suite_name]

    # 전체 통계
    total_episodes, total_successes = 0, 0
    stats = {
        "policy_steps": 0,        # 정책이 action을 낸 스텝 수 (= infer + skip)
        "infer_steps": 0,         # VLA를 실제로 돌린 스텝 수
        "skip_steps": 0,          # ridge regression으로 대체한 스텝 수
        "infer_time": 0.0,        # VLA 호출 총 시간 (get_action, 토큰 선택 포함)
        "skip_time": 0.0,         # skip 예측 총 시간
        "reused_tokens": 0,       # 재사용 후보 토큰 수 합 (레이어 비율 적용 전)
        "cache_steps": 0,         # 재사용이 실제로 걸린 추론 스텝 수
        "episode_time": 0.0,      # 에피소드 wall-clock (sim 포함)
    }

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            done = False
            replay_images = []
            replay_images_heatmap = []
            last_caches = None       # 마지막 추론에서 나온 {past_key_values, attentions}
            last_infer_img = None    # last_caches를 만든 프레임 (VLA-Cache 비교 기준)
            scheduler.reset()
            ep_infer_time, ep_reused = 0.0, 0

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            ep_start = time.time()

            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)
                    replay_images.append(img)

                    if cfg.use_step_skip and scheduler.should_skip():
                        # ===== SKIP PATH: 모델 호출 X, KV 캐시 / 기준 프레임 유지 =====
                        s = time.perf_counter()
                        action = scheduler.predict()
                        stats["skip_time"] += time.perf_counter() - s
                        scheduler.record(action, generated=False)
                        replay_images_heatmap.append(img)
                    else:
                        # ===== INFERENCE PATH: VLA(+VLA-Cache) 실행 =====
                        # 비교 기준 = last_caches를 만든 프레임 (첫 추론이면 자기 자신)
                        prev_img = last_infer_img if last_infer_img is not None else img
                        observation = {
                            "full_image": img,
                            "prev_image": prev_img,
                            "state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                        }

                        cuda_sync()
                        s = time.perf_counter()
                        action, last_caches, result_image = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            last_caches=last_caches,
                        )
                        cuda_sync()
                        ep_infer_time += time.perf_counter() - s

                        # 이번 추론에서 재사용된 토큰 수 (get_vla_action이 config에 기록)
                        lm_cfg = model.language_model.config
                        if cfg.use_vla_cache and getattr(lm_cfg, "proportion_attn_var", None) is not None:
                            patches = getattr(lm_cfg, "reusable_patches", None)
                            if patches is not None:
                                ep_reused += len(patches)
                                stats["cache_steps"] += 1

                        last_infer_img = img
                        # normalize_gripper_action은 in-place로 바꾸므로 반드시 복사본을 저장
                        scheduler.record(np.copy(action), generated=True)
                        replay_images_heatmap.append(result_image)

                    action = np.array(action, dtype=np.float64, copy=True)

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] flip gripper sign back (-1 = open, +1 = close)
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    traceback.print_exc()
                    log_file.write(f"Caught exception: {e}\n{traceback.format_exc()}\n")
                    break

            ep_time = time.time() - ep_start
            task_episodes += 1
            total_episodes += 1

            # 에피소드 통계
            n_inf, n_skip = scheduler.episode_infers, scheduler.episode_skips
            n_pol = n_inf + n_skip
            stats["policy_steps"] += n_pol
            stats["infer_steps"] += n_inf
            stats["skip_steps"] += n_skip
            stats["infer_time"] += ep_infer_time
            stats["reused_tokens"] += ep_reused
            stats["episode_time"] += ep_time

            ep_skip_rate = n_skip / max(n_pol, 1) * 100
            ep_infer_ms = ep_infer_time / max(n_inf, 1) * 1000
            ep_msg = (
                f"[episode] success={done} policy_steps={n_pol} infer={n_inf} skip={n_skip} "
                f"skip_rate={ep_skip_rate:.1f}% avg_infer={ep_infer_ms:.1f}ms "
                f"avg_reused_tokens={ep_reused / max(n_inf, 1):.1f} ep_time={ep_time:.1f}s"
            )

            if cfg.save_video:
                save_rollout_video(
                    replay_images_heatmap, total_episodes, success=done, task_description=task_description, log_file=log_file
                )

            print(ep_msg)
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(ep_msg + "\n")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

            if cfg.use_wandb:
                wandb.log({
                    "episode/skip_rate": ep_skip_rate,
                    "episode/avg_infer_ms": ep_infer_ms,
                    "cumulative_success_rate": total_successes / total_episodes,
                })

        # Log task results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log({
                f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                f"num_episodes/{task_description}": task_episodes,
            })

    # ===== 최종 요약 =====
    P, I = max(stats["policy_steps"], 1), max(stats["infer_steps"], 1)
    summary = {
        "mode": mode,
        "success_rate": total_successes / max(total_episodes, 1),
        "episodes": total_episodes,
        "policy_steps": stats["policy_steps"],
        "infer_steps": stats["infer_steps"],
        "skip_steps": stats["skip_steps"],
        "skip_rate": stats["skip_steps"] / P,
        "avg_infer_latency_ms": stats["infer_time"] / I * 1000,                               # VLA 1회 호출
        "avg_policy_latency_ms": (stats["infer_time"] + stats["skip_time"]) / P * 1000,       # 스텝당 (skip 포함)
        "avg_reused_tokens_per_infer": stats["reused_tokens"] / I,
        "cache_hit_infer_ratio": stats["cache_steps"] / I,
        "policy_hz": P / max(stats["infer_time"] + stats["skip_time"], 1e-9),
        "total_episode_time_s": stats["episode_time"],
    }
    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    log_file.write("\n===== SUMMARY =====\n" + json.dumps(summary, indent=2) + "\n")
    log_file.close()

    with open(os.path.join(cfg.local_log_dir, run_id + "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if cfg.use_wandb:
        wandb.log({f"summary/{k}": v for k, v in summary.items() if isinstance(v, (int, float))})
        wandb.log({
            "success_rate/total": summary["success_rate"],
            "num_episodes/total": total_episodes,
        })
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()