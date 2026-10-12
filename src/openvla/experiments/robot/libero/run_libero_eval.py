"""
run_libero_eval.py  (VLA-Cache + SP-VLA Step Skipping)

VLA-Cache(토큰 KV 재사용) 평가 스크립트에 SP-VLA의 action 재사용(step skipping)을 추가한 버전.

  - use_vla_cache     : 모델을 돌릴 때 정적 토큰의 KV를 재사용 (VLA-Cache)
  - use_step_skip     : 평면 이동 구간에서는 모델을 건너뛰고 ridge regression으로 action 예측 (SP-VLA)
  - output_attentions : 캐시를 끈 기준선에서만 의미 있음 (False = B0, True = B1)
  - fork_profiling    : transformers 포크의 LlamaModel.forward 안 synchronize / CUDA 이벤트 / print 사용 여부
                        (기본 False = 끔. 이 동기화가 측정 latency를 부풀리기 때문)

측정 지표 (summary.json)
  [정확도]   success_rate, per_task_success
  [action 재사용]
             policy_steps / infer_steps / skip_steps / skip_rate   모델 호출 수와 건너뛴 수
  [KV 재사용]
             cache_hit_infers            재사용이 실제로 적용된 추론 횟수
             avg_reuse_candidates        추론당 재사용 후보 토큰 수 (이미지 토큰 256개 중)
             avg_reused_tokens_by_layer  레이어별 실제 재사용 토큰 수의 평균 (레이어 비율 적용 후)
             vision_token_compute_skip_ratio
                                         이미지 토큰 x 레이어 계산 중 생략한 비율
  [속도]     avg_infer_latency_ms        VLA 1회 호출 (get_action 앞뒤 동기화 + perf_counter)
             avg_prefill_ms              LLM 프리필 (CUDA 이벤트, 추가 동기화 없음)
             avg_policy_latency_ms       skip 포함 스텝당 평균
             policy_hz                   1000 / avg_policy_latency_ms
  [기타]     peak_gpu_mem_gb, total_episode_time_s

Usage:
    COMMON="--pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial --save_video False"
    python experiments/robot/libero/run_libero_eval.py $COMMON --use_vla_cache False --output_attentions False --use_step_skip False  # B0
    python experiments/robot/libero/run_libero_eval.py $COMMON --use_vla_cache False --output_attentions True  --use_step_skip False  # B1
    python experiments/robot/libero/run_libero_eval.py $COMMON --use_vla_cache True  --use_step_skip False                            # V
    python experiments/robot/libero/run_libero_eval.py $COMMON --use_vla_cache False --output_attentions False --use_step_skip True   # B0+skip
    python experiments/robot/libero/run_libero_eval.py $COMMON --use_vla_cache True  --use_step_skip True                             # V+skip
"""

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

NUM_VISION_TOKENS = 256     # OpenVLA: 224x224 이미지 → 16x16 패치


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Acceleration switches
    #################################################################################################################
    use_vla_cache: bool = True                       # VLA-Cache: 정적 토큰 KV 재사용
    use_step_skip: bool = True                       # SP-VLA: action 재사용 (step skipping)
    output_attentions: bool = True                   # 캐시를 끈 경우만 의미 있음 (False=B0, True=B1). openvla_utils.py 수정본 필요
    fork_profiling: bool = False                     # transformers 포크 안의 synchronize/print 사용 (False 권장)

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


def disable_fork_profiling():
    """
    VLA-Cache transformers 포크의 LlamaModel.forward에는 측정용 코드가 들어 있다.
      - forward 시작마다 torch.cuda.synchronize()          (프리필 1회 + 디코드 6회 = 스텝당 7회)
      - 프리필 끝에 torch.cuda.synchronize() + CUDA 이벤트 시간 계산 + print
    CPU가 GPU를 기다리게 만들어 측정 latency를 부풀리므로, 이 모듈 안에서만 해당 호출을 무력화한다.
    (포크 파일은 수정하지 않는다. 다른 모듈의 torch.cuda.synchronize는 그대로 동작한다.)
    """
    import transformers.models.llama.modeling_llama as ml

    class _NoopEvent:
        def __init__(self, *args, **kwargs):
            pass

        def record(self, *args, **kwargs):
            pass

        def elapsed_time(self, other):
            return 0.0

    class _CudaProxy:
        Event = _NoopEvent

        @staticmethod
        def synchronize(*args, **kwargs):
            pass

        def __getattr__(self, name):
            return getattr(torch.cuda, name)

    class _TorchProxy:
        cuda = _CudaProxy()

        def __getattr__(self, name):
            return getattr(torch, name)

    ml.torch = _TorchProxy()
    ml.print = lambda *args, **kwargs: None


class PrefillTimer:
    """LLM 프리필(입력 길이 > 1인 forward)을 CUDA 이벤트로 잰다. 이벤트만 기록하고 동기화는 하지 않는다."""

    def __init__(self, vla):
        self.enabled = torch.cuda.is_available()
        self.pairs = []
        self._start = None
        if self.enabled:
            llm = vla.language_model.model
            llm.register_forward_pre_hook(self._pre, with_kwargs=True)
            llm.register_forward_hook(self._post, with_kwargs=True)

    def _pre(self, m, args, kwargs):
        x = kwargs.get("inputs_embeds")
        if x is None:
            x = kwargs.get("input_ids")
        if x is not None and x.shape[1] > 1:
            self._start = torch.cuda.Event(enable_timing=True)
            self._start.record()

    def _post(self, m, args, kwargs, out):
        if self._start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.pairs.append((self._start, end))
            self._start = None

    def pop_ms(self):
        """스텝이 끝나고 이미 동기화된 뒤에 호출한다."""
        ms = sum(s.elapsed_time(e) for s, e in self.pairs) if self.pairs else 0.0
        self.pairs = []
        return ms


def reused_tokens_by_layer(vla):
    """
    이번 추론에서 레이어별로 실제 재사용된(계산을 생략한) 이미지 토큰 수.
    transformers 포크 LlamaModel.forward의 규칙을 그대로 따른다:
      - 재사용은 pruning_loc 레이어(2, 6, 9, 11)에서만 적용
      - 그 레이어의 개수 = max(1, int(proportion[layer] * 후보 수)), 이전보다 크거나 같을 때만 늘어남
      - 한 번 생략한 토큰은 이후 레이어에서도 계속 생략
    재사용이 없으면 None.
    """
    lm_cfg = vla.language_model.config
    prop = getattr(lm_cfg, "proportion_attn_var", None)
    patches = getattr(lm_cfg, "reusable_patches", None)
    if prop is None or patches is None:
        return None
    prop = prop.tolist() if hasattr(prop, "tolist") else list(prop)
    llm = vla.language_model.model
    pruning_loc = set(getattr(llm, "pruning_loc", [2, 6, 9, 11]))
    n_cand = len(patches)
    cur, per_layer = 0, []
    for layer_idx in range(len(llm.layers)):
        if layer_idx in pruning_loc and layer_idx < len(prop):
            cur = max(cur, max(1, int(prop[layer_idx] * n_cand)))
        per_layer.append(cur)
    return n_cand, per_layer


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

    # 포크 안의 측정용 동기화 끄기
    if not cfg.fork_profiling:
        disable_fork_profiling()

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
    prefill_timer = PrefillTimer(model)
    num_layers = len(model.language_model.model.layers)

    # 방법 이름: B0 = 캐시 끔+어텐션 끔, B1 = 캐시 끔+어텐션 켬, V = VLA-Cache
    mode = "V" if cfg.use_vla_cache else ("B1" if cfg.output_attentions else "B0")
    if cfg.use_step_skip:
        mode += "+skip"

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{mode.replace('+', '_')}-{DATE_TIME}"
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
    per_task_success = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stats = {
        "policy_steps": 0,        # 정책이 action을 낸 스텝 수 (= infer + skip)
        "infer_steps": 0,         # VLA를 실제로 돌린 스텝 수
        "skip_steps": 0,          # ridge regression으로 대체한 스텝 수 (action 재사용)
        "infer_time": 0.0,        # VLA 호출 총 시간 (get_action, 토큰 선택 포함)
        "skip_time": 0.0,         # skip 예측 총 시간
        "prefill_time_ms": 0.0,   # LLM 프리필 총 시간 (CUDA 이벤트)
        "cache_hit_infers": 0,    # KV 재사용이 실제로 적용된 추론 수
        "reuse_candidates": 0,    # 재사용 후보 토큰 수 합
        "reused_token_layers": 0, # 레이어별 실제 재사용 토큰 수의 합 (토큰 x 레이어)
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
            ep_infer_time, ep_cache_hits, ep_candidates, ep_reused_tl = 0.0, 0, 0, 0

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

                        # 스텝 latency: 시작 전 / 끝난 후 1번씩만 동기화 (이 값이 기준 측정)
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
                        stats["prefill_time_ms"] += prefill_timer.pop_ms()

                        # 이번 추론의 KV 재사용 (get_vla_action이 config에 기록)
                        if cfg.use_vla_cache:
                            reuse = reused_tokens_by_layer(model)
                            if reuse is not None:
                                n_cand, per_layer = reuse
                                ep_cache_hits += 1
                                ep_candidates += n_cand
                                ep_reused_tl += sum(per_layer)

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
            stats["cache_hit_infers"] += ep_cache_hits
            stats["reuse_candidates"] += ep_candidates
            stats["reused_token_layers"] += ep_reused_tl
            stats["episode_time"] += ep_time

            ep_skip_rate = n_skip / max(n_pol, 1) * 100
            ep_infer_ms = ep_infer_time / max(n_inf, 1) * 1000
            ep_msg = (
                f"[episode] success={done} steps={n_pol} infer={n_inf} action_skip={n_skip} "
                f"({ep_skip_rate:.1f}%) | kv_reuse_infers={ep_cache_hits} "
                f"avg_candidates={ep_candidates / max(ep_cache_hits, 1):.1f} "
                f"avg_reused_by_layer={ep_reused_tl / max(ep_cache_hits, 1) / num_layers:.1f} "
                f"| avg_infer={ep_infer_ms:.1f}ms ep_time={ep_time:.1f}s"
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
        per_task_success[task_description] = float(task_successes) / float(task_episodes)
        if cfg.use_wandb:
            wandb.log({
                f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                f"num_episodes/{task_description}": task_episodes,
            })

    # ===== 최종 요약 =====
    P, I = max(stats["policy_steps"], 1), max(stats["infer_steps"], 1)
    H = max(stats["cache_hit_infers"], 1)
    avg_policy_ms = (stats["infer_time"] + stats["skip_time"]) / P * 1000
    summary = {
        "mode": mode,
        "fork_profiling": cfg.fork_profiling,

        # 정확도
        "success_rate": total_successes / max(total_episodes, 1),
        "episodes": total_episodes,

        # action 재사용 (SP-VLA)
        "policy_steps": stats["policy_steps"],
        "infer_steps": stats["infer_steps"],
        "skip_steps": stats["skip_steps"],
        "skip_rate": stats["skip_steps"] / P,

        # KV 재사용 (VLA-Cache)
        "cache_hit_infers": stats["cache_hit_infers"],
        "avg_reuse_candidates": stats["reuse_candidates"] / H,
        "avg_reused_tokens_by_layer": stats["reused_token_layers"] / H / num_layers,
        "vision_token_compute_skip_ratio": stats["reused_token_layers"] / (I * NUM_VISION_TOKENS * num_layers),

        # 속도
        "avg_infer_latency_ms": stats["infer_time"] / I * 1000,      # VLA 1회 호출
        "avg_prefill_ms": stats["prefill_time_ms"] / I,              # LLM 프리필
        "avg_policy_latency_ms": avg_policy_ms,                      # 스텝당 (skip 포함)
        "policy_hz": 1000 / avg_policy_ms if avg_policy_ms > 0 else None,

        # 기타
        "total_episode_time_s": stats["episode_time"],
    }
    if torch.cuda.is_available():
        summary["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    summary["per_task_success"] = per_task_success

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
