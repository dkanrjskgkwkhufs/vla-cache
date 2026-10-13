"""
run_libero_eval.py  (VLA-Cache + SP-VLA Step Skipping)

조건 (cond)
  C0  : 기본 (캐시 끔, 어텐션 출력 끔)          --use_vla_cache False --output_attentions False --use_step_skip False
  CS  : 건너뛰기만 (C0 + action 재사용)         --use_vla_cache False --output_attentions False --use_step_skip True
  B1  : 캐시 끔, 어텐션 출력 켬 (V의 비교 상대)  --use_vla_cache False --output_attentions True  --use_step_skip False
  V   : VLA-Cache                              --use_vla_cache True  --use_step_skip False
  VS  : VLA-Cache + action 재사용              --use_vla_cache True  --use_step_skip True
  (C0 = KV 캐시 실험 문서의 B0)

출력 (experiments/logs/)
  <run_id>.txt           사람이 읽는 로그
  <run_id>_steps.csv     스텝(행동 1개)마다 1줄
  <run_id>_summary.json  요약 지표

스텝 CSV 열
  cond, task_id, episode, step, step_type(infer/skip)
  t_step_ms      행동 1개를 내는 전체 시간 = 전처리 + select + generate  (KV 문서의 "스텝 전체")
  t_total_ms     t_step_ms - t_preproc_ms  (SP-VLA 계획의 t_total: 이미지 전처리 제외)
  t_preproc_ms   CPU 이미지 전처리 (center crop + processor)
  t_select_ms    재사용 토큰 고르기 (V, VS)
  t_vision_ms    이미지 인코더 + projector  (GPU, CUDA 이벤트)
  t_prefill_ms   언어모델 프리필            (GPU, CUDA 이벤트)
  t_decode_ms    언어모델 디코드 합계        (GPU, CUDA 이벤트), n_decode = 디코드 forward 횟수
  t_generate_ms  모델 실행 전체 (비전 + 언어모델 + action 디코딩)
  t_fit_ms       skip 때 직선(ridge) 예측 시간
  t_obs_ms       시뮬레이터 이미지 → 224x224 (get_libero_image). t_step에는 포함되지 않음
  n_reuse_cand   재사용 후보 토큰 수 (이미지 토큰 256개 중)
  n_reused_avg   레이어별 실제 재사용 토큰 수의 32레이어 평균
  prefill_tflops 언어모델 프리필 연산량 (transformers 포크가 계산)
  token_match    B1과 같은 행동 토큰 비율 (--shadow_compare True 일 때만)
  success, episode_len   에피소드 마지막 줄에만
  skip 스텝은 모델을 돌리지 않으므로 preproc/select/vision/prefill/decode/generate가 0

시간 통계는 KV 캐시 실험 문서와 같은 정의: 에피소드별 첫 10스텝 제외, 중앙값.
"""

import copy
import csv
import json
import math
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

CSV_FIELDS = [
    "cond", "task_id", "episode", "step", "step_type",
    "t_step_ms", "t_total_ms", "t_preproc_ms", "t_select_ms",
    "t_vision_ms", "t_prefill_ms", "t_decode_ms", "n_decode", "t_generate_ms",
    "t_fit_ms", "t_obs_ms",
    "n_reuse_cand", "n_reused_avg", "prefill_tflops", "token_match",
    "success", "episode_len",
]


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Conditions
    #################################################################################################################
    use_vla_cache: bool = True                       # VLA-Cache: 정적 토큰 KV 재사용
    use_step_skip: bool = True                       # SP-VLA: action 재사용 (step skipping)
    output_attentions: bool = True                   # 캐시를 끈 경우만 의미 있음 (False=C0, True=B1)
    fork_profiling: bool = False                     # transformers 포크 안의 synchronize/print 사용 (False 권장)

    # Step skipping 하이퍼파라미터 (None이면 SKIP_PARAMS의 suite 기본값 사용)
    z_xy_rate_skip: Optional[float] = None           # max|z| / max(|x|,|y|) < 이 값이어야 skip
    z_max_skip: Optional[float] = None               # |z| < 이 값이어야 skip
    step_skip: Optional[int] = None                  # 최근 몇 개 action이 위 조건을 만족해야 하는지
    action_buffer_size: int = 6                      # action 버퍼 크기 (SP-VLA: 6)
    min_generate_ratio: float = 4 / 6                # 버퍼 내 VLA 생성 action 비율 하한 (SP-VLA: 4/6)

    #################################################################################################################
    # Measurement
    #################################################################################################################
    timing_warmup_steps: int = 10                    # 시간 통계에서 제외할 에피소드별 첫 정책 스텝 수 (KV 문서와 동일)
    shadow_compare: bool = False                     # V/VS에서 추론마다 같은 입력으로 B1을 한 번 더 돌려 행동 토큰 일치율 측정 (약 2배 느림)
    verbose: bool = True                             # 공개 코드의 스텝별 print (">> VLA-Cache ...") 유지

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


# =====================================================================================================================
# SP-VLA step skipping
# =====================================================================================================================
class ActionScheduler:
    """
    SP-VLA step skipping (SP-VLA prismatic/extern/hf/modeling_prismatic.py 의 compressor 로직을 모델 밖으로 이식).
    원본과 같은 결정을 내리는지는 check_skip_equivalence.py 로 확인할 수 있다.
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

    # --- SP-VLA save_actions ---
    def record(self, action, generated: bool):
        self.actions.append(np.array(action, dtype=np.float64, copy=True))
        self.types.append("generate" if generated else "fit")
        if generated:
            self.num_generate += 1
        if len(self.actions) > self.buffer_size:
            self.actions.pop(0)
            if self.types.pop(0) == "generate":
                self.num_generate -= 1

    # --- SP-VLA skip 조건 ---
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

    # --- SP-VLA fit_next_action (ridge regression) ---
    def predict(self) -> np.ndarray:
        total = np.array(self.actions)
        Y = total[:, :3]                                         # (N, 3) xyz translation
        X = np.arange(len(Y)).reshape(-1, 1)
        X = np.hstack([X, np.ones_like(X)])                      # (N, 2)
        beta = np.linalg.solve(X.T @ X + 1e-4 * np.eye(2), X.T @ Y)   # (2, 3)
        next_xyz = np.clip(beta[0] * len(Y) + beta[1], -0.9, 0.9)
        # 증가량이 과거 최대 증가량을 넘지 않도록 제한
        delta_max = np.max(np.abs(np.diff(Y, axis=0)), axis=0)
        last_xyz = Y[-1]
        next_xyz = last_xyz + np.clip(next_xyz - last_xyz, -delta_max, delta_max)
        # 회전 + gripper는 직전 action 그대로 (SP-VLA 코드 동작)
        return np.hstack([next_xyz, total[-1, 3:]])


# =====================================================================================================================
# 측정 도구
# =====================================================================================================================
def disable_fork_profiling():
    """
    VLA-Cache transformers 포크의 LlamaModel.forward에는 측정용 코드가 들어 있다.
      - forward 시작마다 torch.cuda.synchronize()          (프리필 1회 + 디코드 6회 = 스텝당 7회)
      - 프리필 끝에 torch.cuda.synchronize() + CUDA 이벤트 시간 계산 + print
    CPU가 GPU를 기다리게 만들어 측정 latency를 부풀리므로, 이 모듈 안에서만 해당 호출을 무력화한다.
    (포크 파일은 수정하지 않는다. 다른 모듈의 torch.cuda.synchronize는 그대로 동작한다.
     포크의 FLOPs 계산은 순수 파이썬이라 영향 없이 계속 동작한다.)
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


class StageTimer:
    """
    GPU 구간을 CUDA 이벤트로 잰다. hook은 이벤트를 기록만 하고 동기화하지 않는다.
    스텝이 끝나 이미 동기화된 뒤에 collect()로 한 번에 읽는다.
      vision  : vision_backbone 시작 ~ projector 끝
      prefill : LlamaModel forward 중 입력 길이 > 1
      decode  : LlamaModel forward 중 입력 길이 == 1 (합계, 횟수)
    """

    def __init__(self, vla):
        self.enabled = torch.cuda.is_available()
        self.events = []       # (stage, start, end)
        self._open = {}
        if not self.enabled:
            return
        llm = vla.language_model.model
        vla.vision_backbone.register_forward_pre_hook(lambda m, a: self._start("vision"))
        vla.projector.register_forward_hook(lambda m, a, o: self._end("vision"))
        llm.register_forward_pre_hook(self._llm_pre, with_kwargs=True)
        llm.register_forward_hook(self._llm_post, with_kwargs=True)

    def _event(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _start(self, stage):
        if self.enabled:
            self._open[stage] = self._event()

    def _end(self, stage):
        if self.enabled and stage in self._open:
            self.events.append((stage, self._open.pop(stage), self._event()))

    def _llm_pre(self, m, args, kwargs):
        x = kwargs.get("inputs_embeds")
        if x is None:
            x = kwargs.get("input_ids")
        if x is None and args:
            x = args[0]
        self._start("prefill" if (x is not None and x.shape[1] > 1) else "decode")

    def _llm_post(self, m, args, kwargs, out):
        self._end("prefill" if "prefill" in self._open else "decode")

    def reset(self):
        self.events, self._open = [], {}

    def collect(self):
        out = {"vision": 0.0, "prefill": 0.0, "decode": 0.0, "n_decode": 0}
        for stage, s, e in self.events:
            out[stage] += s.elapsed_time(e)
            if stage == "decode":
                out["n_decode"] += 1
        self.reset()
        return out


def reused_tokens_by_layer(vla):
    """
    이번 추론에서 레이어별로 실제 재사용된(계산을 생략한) 이미지 토큰 수.
    transformers 포크 LlamaModel.forward 규칙을 그대로 따른다:
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


def build_observation(obs, img, prev_img):
    return {
        "full_image": img,
        "prev_image": prev_img,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }


def shadow_compare(cfg, model, processor, timer, llm_core, obs, img, task_description, action):
    """
    같은 입력으로 B1(캐시 끔, 어텐션 출력 켬)을 한 번 더 돌려 행동 토큰 일치율을 잰다 (KV 문서의 "행동 토큰 일치율").
    본 실행의 KV 캐시·기준 프레임·시간·FLOPs 통계는 건드리지 않는다.
    역정규화는 토큰마다 서로 다른 값을 주므로, action 값이 같으면 토큰도 같다.
    """
    shadow_cfg = copy.copy(cfg)
    shadow_cfg.use_vla_cache = False
    shadow_cfg.output_attentions = True
    shadow_cfg.verbose = False
    saved = {k: getattr(llm_core, k, None) for k in ("all_FLOPs", "total_cuda_time", "num_forward")}
    saved_timing = getattr(model, "_step_timing", None)
    timer.enabled = False
    try:
        ref_action, _, _ = get_action(shadow_cfg, model, build_observation(obs, img, img), task_description,
                                      processor=processor, last_caches=None)
    finally:
        timer.enabled = torch.cuda.is_available()
        timer.reset()
        for k, v in saved.items():
            if v is not None:
                setattr(llm_core, k, v)
        model._step_timing = saved_timing
    return float(np.mean(np.asarray(action, dtype=np.float64) == np.asarray(ref_action, dtype=np.float64)))


def _median(xs):
    return float(np.median(xs)) if len(xs) else None


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


def make_cond(cfg):
    if cfg.use_vla_cache:
        base = "V"
    else:
        base = "B1" if cfg.output_attentions else "C0"
    if not cfg.use_step_skip:
        return base
    return {"C0": "CS", "B1": "B1S", "V": "VS"}[base]


# =====================================================================================================================
# 평가
# =====================================================================================================================
@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name

    # Step skipping 하이퍼파라미터
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
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"
    processor = get_processor(cfg) if cfg.model_family == "openvla" else None

    # 실제로 쓰이는 어텐션 구현 (sdpa / eager / flash_attention_2)
    attn_impl = getattr(model.language_model.config, "_attn_implementation", None)
    print(f"[attn] language model attention implementation: {attn_impl}")

    scheduler = ActionScheduler(cfg.z_xy_rate_skip, cfg.z_max_skip, cfg.step_skip,
                                cfg.action_buffer_size, cfg.min_generate_ratio)
    timer = StageTimer(model)
    llm_core = model.language_model.model
    num_layers = len(llm_core.layers)
    cond = make_cond(cfg)
    do_shadow = cfg.shadow_compare and cfg.use_vla_cache

    # Logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{cond}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    log_path = os.path.join(cfg.local_log_dir, run_id + ".txt")
    csv_path = os.path.join(cfg.local_log_dir, run_id + "_steps.csv")
    summary_path = os.path.join(cfg.local_log_dir, run_id + "_summary.json")
    log_file = open(log_path, "w")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    csv_writer.writeheader()
    print(f"Logging to: {log_path}\nStep CSV : {csv_path}")
    log_file.write(f"Config: {json.dumps({k: str(v) for k, v in vars(cfg).items()}, ensure_ascii=False)}\n")
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    resize_size = get_image_resize_size(cfg)
    max_steps = MAX_STEPS[cfg.task_suite_name]
    log_file.write(f"Task suite: {cfg.task_suite_name} | cond: {cond} | attn_implementation: {attn_impl}\n")

    # 집계용
    total_episodes, total_successes = 0, 0
    per_task_success = {}
    all_rows = []                    # 요약 계산용 (CSV와 같은 내용)
    total_env_steps, total_episode_time = 0, 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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

            t = 0
            step = 0                 # 정책 스텝 번호 (대기 스텝 제외, 0부터)
            done = False
            ep_rows = []
            replay_images = []
            last_caches = None       # 마지막 추론에서 나온 {past_key_values, attentions}
            last_infer_img = None    # last_caches를 만든 프레임 (VLA-Cache 비교 기준)
            scheduler.reset()
            ep_start = time.time()

            while t < max_steps + cfg.num_steps_wait:
                try:
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    s = time.perf_counter()
                    img = get_libero_image(obs, resize_size)
                    t_obs = (time.perf_counter() - s) * 1000

                    row = {k: None for k in CSV_FIELDS}
                    row.update(cond=cond, task_id=task_id, episode=episode_idx, step=step, t_obs_ms=t_obs)

                    if cfg.use_step_skip and scheduler.should_skip():
                        # ===== SKIP: 모델 호출 X, KV 캐시 / 기준 프레임 유지 =====
                        s = time.perf_counter()
                        action = scheduler.predict()
                        t_fit = (time.perf_counter() - s) * 1000
                        scheduler.record(action, generated=False)
                        replay_images.append(img)
                        row.update(step_type="skip", t_step_ms=t_fit, t_total_ms=t_fit, t_fit_ms=t_fit,
                                   t_preproc_ms=0.0, t_select_ms=0.0, t_vision_ms=0.0, t_prefill_ms=0.0,
                                   t_decode_ms=0.0, n_decode=0, t_generate_ms=0.0,
                                   n_reuse_cand=0, n_reused_avg=0.0, prefill_tflops=0.0)
                    else:
                        # ===== INFER: VLA(+VLA-Cache). 비교 기준 = last_caches를 만든 프레임 =====
                        observation = build_observation(obs, img, last_infer_img if last_infer_img is not None else img)
                        flops_before = getattr(llm_core, "all_FLOPs", 0)

                        cuda_sync()
                        timer.reset()
                        s = time.perf_counter()
                        action, last_caches, result_image = get_action(
                            cfg, model, observation, task_description, processor=processor, last_caches=last_caches
                        )
                        cuda_sync()
                        t_step = (time.perf_counter() - s) * 1000

                        gpu = timer.collect()
                        tm = getattr(model, "_step_timing", {}) or {}
                        t_pre = tm.get("preproc_ms", 0.0)

                        n_cand, n_reused_avg = 0, 0.0
                        if cfg.use_vla_cache:
                            reuse = reused_tokens_by_layer(model)
                            if reuse is not None:
                                n_cand = reuse[0]
                                n_reused_avg = sum(reuse[1]) / num_layers

                        row.update(step_type="infer", t_step_ms=t_step, t_total_ms=t_step - t_pre, t_fit_ms=0.0,
                                   t_preproc_ms=t_pre, t_select_ms=tm.get("select_ms", 0.0),
                                   t_vision_ms=gpu["vision"], t_prefill_ms=gpu["prefill"],
                                   t_decode_ms=gpu["decode"], n_decode=gpu["n_decode"],
                                   t_generate_ms=tm.get("generate_ms", 0.0),
                                   n_reuse_cand=n_cand, n_reused_avg=n_reused_avg,
                                   prefill_tflops=(getattr(llm_core, "all_FLOPs", 0) - flops_before) * 1e-12)

                        if do_shadow:
                            row["token_match"] = shadow_compare(cfg, model, processor, timer, llm_core, obs, img,
                                                                task_description, action)

                        last_infer_img = img
                        scheduler.record(np.copy(action), generated=True)   # gripper 변환 전 값 저장
                        replay_images.append(result_image)

                    ep_rows.append(row)
                    step += 1

                    action = np.array(action, dtype=np.float64, copy=True)
                    action = normalize_gripper_action(action, binarize=True)
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
            total_env_steps += t
            total_episode_time += ep_time
            task_episodes += 1
            total_episodes += 1

            # 에피소드 마지막 줄에 success / episode_len
            if ep_rows:
                ep_rows[-1]["success"] = int(bool(done))
                ep_rows[-1]["episode_len"] = len(ep_rows)
            for r in ep_rows:
                csv_writer.writerow({k: (_r(v) if isinstance(v, float) else v) for k, v in r.items()})
            csv_file.flush()
            all_rows.extend(ep_rows)

            if cfg.save_video:
                save_rollout_video(replay_images, total_episodes, success=done,
                                   task_description=task_description, log_file=log_file)

            n_skip = sum(r["step_type"] == "skip" for r in ep_rows)
            inf_ms = [r["t_step_ms"] for r in ep_rows if r["step_type"] == "infer"]
            ep_msg = (f"[episode] task={task_id} ep={episode_idx} success={done} steps={len(ep_rows)} "
                      f"infer={len(ep_rows) - n_skip} skip={n_skip} "
                      f"median_infer_step={_median(inf_ms) or 0:.1f}ms ep_time={ep_time:.1f}s | "
                      f"successes {total_successes}/{total_episodes} ({total_successes / total_episodes * 100:.1f}%)")
            print(ep_msg)
            log_file.write(ep_msg + "\n")
            log_file.flush()
            if cfg.use_wandb:
                wandb.log({"cumulative_success_rate": total_successes / total_episodes})

        per_task_success[task_description] = task_successes / max(task_episodes, 1)
        log_file.write(f"Task success rate: {per_task_success[task_description]}\n")
        log_file.flush()

    csv_file.close()
    summary = summarize(cfg, cond, all_rows, total_episodes, total_successes, per_task_success,
                        total_env_steps, total_episode_time, num_layers)
    summary["attn_implementation"] = attn_impl

    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    log_file.write("\n===== SUMMARY =====\n" + json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    log_file.close()
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if cfg.use_wandb:
        wandb.log({f"summary/{k}": v for k, v in summary.items() if isinstance(v, (int, float))})
        wandb.save(log_path)


def summarize(cfg, cond, rows, total_episodes, total_successes, per_task_success,
              total_env_steps, total_episode_time, num_layers):
    """
    시간: 에피소드별 첫 timing_warmup_steps 스텝 제외.
      *_median  : 모델을 호출한 스텝(infer)의 중앙값  → KV 캐시 문서 표와 같은 정의
      *_mean_all: skip 포함 모든 스텝의 평균         → 두 기법을 합친 실제 처리 속도
    횟수·FLOPs·성공률: 전체 스텝/에피소드.
    """
    w = cfg.timing_warmup_steps
    timed = [r for r in rows if r["step"] >= w]
    inf_t = [r for r in timed if r["step_type"] == "infer"]
    skip_t = [r for r in timed if r["step_type"] == "skip"]
    inf_all = [r for r in rows if r["step_type"] == "infer"]
    n_steps = len(rows)
    n_skip = sum(r["step_type"] == "skip" for r in rows)

    def med(key, rs=inf_t):
        return _r(_median([r[key] for r in rs]))

    p = total_successes / max(total_episodes, 1)
    se = math.sqrt(p * (1 - p) / total_episodes) if total_episodes else None
    step_median = _median([r["t_step_ms"] for r in inf_t])
    step_mean_all = _mean([r["t_step_ms"] for r in timed])
    reuse = [r["n_reuse_cand"] for r in inf_all if r["n_reuse_cand"]]
    flops = [r["prefill_tflops"] for r in inf_all]
    tm = [r["token_match"] for r in rows if r.get("token_match") is not None]

    summary = {
        "cond": cond,
        "episodes": total_episodes,
        "timing_excluded_first_steps": w,
        "fork_profiling": cfg.fork_profiling,

        # 정확도
        "success_rate": _r(p, 4),
        "success_rate_se": _r(se, 4),                                   # 표준오차 sqrt(p(1-p)/n)
        "success_rate_ci95": [_r(max(0.0, p - 1.96 * se), 4), _r(min(1.0, p + 1.96 * se), 4)] if se is not None else None,
        "successes": total_successes,

        # action 재사용 (SP-VLA)
        "policy_steps": n_steps,
        "infer_steps": n_steps - n_skip,
        "skip_steps": n_skip,
        "skip_rate": _r(n_skip / max(n_steps, 1), 4),

        # 시간: 모델 호출 스텝 중앙값 (KV 문서 표와 같은 정의)
        "t_step_ms_median": _r(step_median),                            # 스텝 전체
        "t_total_ms_median": med("t_total_ms"),                         # 전처리 제외
        "t_preproc_ms_median": med("t_preproc_ms"),
        "t_select_ms_median": med("t_select_ms"),
        "t_vision_ms_median": med("t_vision_ms"),
        "t_prefill_ms_median": med("t_prefill_ms"),
        "t_decode_ms_median": med("t_decode_ms"),
        "n_decode_median": med("n_decode"),
        "t_generate_ms_median": med("t_generate_ms"),
        "control_hz": _r(1000 / step_median) if step_median else None,  # 제어 주파수 (모델 호출 스텝 기준)

        # 시간: skip 포함
        "t_fit_ms_median": med("t_fit_ms", skip_t),
        "t_step_ms_mean_all": _r(step_mean_all),
        "t_step_ms_median_all": _r(_median([r["t_step_ms"] for r in timed])),
        "effective_hz": _r(1000 / step_mean_all) if step_mean_all else None,   # 두 기법을 합친 실제 처리 속도
        "t_obs_ms_median": med("t_obs_ms", timed),

        # SP-VLA 논문 방식 Hz: 에피소드 전체 시간(시뮬레이터·전처리·모델 포함, 대기 스텝 포함)
        "episode_level_hz": _r(total_env_steps / total_episode_time) if total_episode_time > 0 else None,
        "total_episode_time_s": _r(total_episode_time, 1),

        # 연산량 (언어모델 프리필, transformers 포크 계산. VLA-Cache 논문과 같은 기준)
        "prefill_tflops_per_infer": _r(_mean(flops), 4),
        "prefill_tflops_per_step": _r(sum(flops) / max(n_steps, 1), 4),    # skip 포함 → SP-VLA의 FLOPs 비교용

        # KV 재사용 (VLA-Cache)
        "cache_hit_infers": len(reuse),
        "reuse_cand_median": _r(_median(reuse)),
        "reuse_cand_min": int(min(reuse)) if reuse else None,
        "reuse_cand_max": int(max(reuse)) if reuse else None,
        "reused_tokens_by_layer_mean": _r(_mean([r["n_reused_avg"] for r in inf_all if r["n_reuse_cand"]])),
        "vision_token_compute_skip_ratio": _r(
            sum(r["n_reused_avg"] for r in inf_all) / (max(len(inf_all), 1) * NUM_VISION_TOKENS), 4),

        # 행동 토큰 일치율 (--shadow_compare True)
        "action_token_match_vs_B1": _r(_mean(tm), 4) if tm else None,

        # 기타
        "peak_gpu_mem_gb": _r(torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else None,
        "per_task_success": per_task_success,
    }
    return summary


if __name__ == "__main__":
    eval_libero()