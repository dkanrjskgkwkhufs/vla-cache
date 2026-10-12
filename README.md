# VLA-Cache + SP-VLA Step Skipping (OpenVLA)

VLA-Cache의 토큰 KV 재사용 위에 SP-VLA의 action 재사용(step skipping)을 결합한 OpenVLA 추론 가속 실험 코드.

- **VLA-Cache**: 모델을 돌릴 때, 이전 프레임과 비교해 변하지 않은 시각 토큰의 KV를 재사용
- **SP-VLA step skipping**: 평면 이동 구간에서는 모델을 호출하지 않고 ridge regression으로 action 예측
- 학습 없이 사용 (체크포인트 가중치 수정 없음)

## 동작 방식

```
매 스텝
 ├─ ActionScheduler.should_skip()
 │    조건: 버퍼(최근 6개) 중 VLA 생성 action ≥ 4개  AND  직전 action이 평면 이동
 │          (|z| / max(|x|,|y|) < z_xy_rate_skip, |z| < z_max_skip)
 │    └─ True  → ridge regression으로 xyz 예측, 회전·gripper는 직전 값 유지
 │               모델 호출 X, last_caches / last_infer_img 유지
 └─ False → VLA 실행 (VLA-Cache)
             prev_image = 마지막으로 모델을 돌린 프레임 (last_caches를 만든 프레임)
             ① 정적 패치 선택 → ② task 관련 토큰 제외 → ③ 레이어별 재사용 비율
             → 결과로 last_caches, last_infer_img 갱신
```

## 원본 VLA-Cache 대비 변경 파일

| 파일 | 변경 내용 |
| --- | --- |
| `src/openvla/experiments/robot/libero/run_libero_eval.py` | `ActionScheduler` 추가, skip 분기, `prev_img`를 마지막 추론 프레임으로 변경, 측정·로깅 추가 |
| `src/openvla/experiments/robot/openvla_utils.py` | 407행: 재사용 후보 토큰이 0개면 `proportion_attn_var=None` (원본에서는 `len(None)` 에러 발생) |

`modeling_prismatic.py`와 transformers 포크는 수정하지 않았다.

## 설치

### Jetson AGX Thor (aarch64)

`pyproject.toml`이 `torch==2.2.0`, `tensorflow==2.15.0`을 고정하므로 그냥 `pip install -e .`를 하면 Jetson용 CUDA torch가 덮어써질 수 있다. 기존 OpenVLA 환경 위에 의존성 없이 설치한다.

```bash
cd vla-cache/src/openvla
pip install -e . --no-deps
pip install "git+https://github.com/siyuhsu/transformers.git@vla-cache-openvla" --no-deps
pip install scikit-image opencv-python
pip install -r experiments/robot/libero/libero_requirements.txt
```

### 설치 확인

```bash
python -c "import torch, transformers; print(torch.cuda.is_available(), transformers.__file__)"
```

`True`가 나오고, transformers 경로가 포크 버전이어야 한다.

## 체크포인트

```bash
cd vla-cache/src/openvla
python vla_cache_scripts/download_model_local.py \
  --model_id openvla/openvla-7b-finetuned-libero-spatial
```

반드시 `src/openvla/checkpoints/` 아래에 둔다. 로드할 때 `./prismatic/`의 `modeling_prismatic.py`가 체크포인트 폴더로 복사되므로, **모든 명령은 `src/openvla`에서 실행**한다.

## 실행

### 1. 스모크 테스트 (태스크당 1회, 10 에피소드)

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --use_vla_cache True --use_step_skip True \
  --num_trials_per_task 1 --save_video False
```

확인할 것:

- 로그에 `Caught exception`이 없음
- 에피소드 로그의 `skip`이 0보다 큼
- SUMMARY의 `avg_reused_tokens_per_infer`가 0보다 큼

### 2. 4가지 조합 비교

```bash
for c in False True; do for s in False True; do
  python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --use_vla_cache $c --use_step_skip $s \
    --save_video False
done; done
```

| 조합 | `--use_vla_cache` | `--use_step_skip` |
| --- | --- | --- |
| baseline | False | False |
| cache only | True | False |
| skip only | False | True |
| cache + skip | True | True |

조합 하나에 500 에피소드(10 태스크 × 50회)다. 먼저 `--num_trials_per_task 10`으로 경향을 확인한 뒤 50회로 돌리는 것을 권장한다.

## 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--use_vla_cache` | True | VLA-Cache KV 재사용 |
| `--use_step_skip` | True | SP-VLA step skipping |
| `--z_xy_rate_skip` | suite별 (spatial 0.4) | skip 조건: |z| / max(|x|,|y|) 상한 |
| `--z_max_skip` | suite별 (spatial 0.3) | skip 조건: |z| 상한 |
| `--step_skip` | suite별 (spatial 1) | 최근 몇 개 action이 조건을 만족해야 하는지 |
| `--action_buffer_size` | 6 | action 버퍼 크기 |
| `--min_generate_ratio` | 0.667 | 버퍼 내 VLA 생성 action 비율 하한 |
| `--num_trials_per_task` | 50 | 태스크당 에피소드 수 |
| `--save_video` | True | 에피소드 영상 저장 (latency 측정 시 False 권장) |
| `--seed` | 7 | 랜덤 시드 |

suite별 skip 기본값은 SP-VLA `run_libero.py`의 값을 그대로 사용한다.

## 출력

`experiments/logs/` 아래에 조합별로 생성된다.

- `EVAL-<suite>-openvla-cache{0|1}-skip{0|1}-<시간>.txt`: 에피소드별 로그
- `EVAL-...-<시간>_summary.json`: 최종 지표

### summary.json 지표

| 키 | 의미 |
| --- | --- |
| `success_rate` | 전체 성공률 |
| `per_task_success` | 태스크별 성공률 |
| `skip_rate` | 전체 정책 스텝 중 skip 비율 |
| `policy_steps` / `infer_steps` / `skip_steps` | 정책 스텝 / VLA 호출 / skip 횟수 |
| `avg_infer_latency_ms` | VLA 1회 호출 wall-clock (전처리·토큰 선택·생성 포함, GPU 동기화) |
| `avg_policy_latency_ms` | skip 포함 스텝당 평균 latency |
| `policy_hz` | 1 / `avg_policy_latency_ms` |
| `llm_prefill_cuda_ms_avg` | LLM backbone prefill의 CUDA 이벤트 latency (transformers 포크 측정) |
| `llm_prefill_tflops_avg` | VLA 1회 호출당 LLM prefill TFLOPs (포크 측정) |
| `llm_prefill_tflops_per_policy_step` | skip 포함 스텝당 평균 TFLOPs |
| `avg_reused_tokens_per_infer` | 추론당 재사용 후보 토큰 수 (레이어별 비율 적용 전) |
| `cache_hit_infer_ratio` | 재사용이 적용된 추론 비율 |
| `peak_gpu_mem_gb` | 최대 GPU 메모리 |
| `total_episode_time_s` | 시뮬레이션 포함 전체 시간 |

## 이 코드로 측정되지 않는 것

- **전력·에너지**: Thor에서는 `tegrastats`를 별도로 실행해 기록해야 한다.
- **레이어별 실제 재사용 토큰 수**: `avg_reused_tokens_per_infer`는 레이어 비율 적용 전 후보 수다.
- **비전 인코더·디코딩 단계의 FLOPs**: 포크의 FLOPs는 LLM prefill만 센다 (VLA-Cache 논문과 같은 기준).

## 주의

- `flash_attention_2`를 켜면 안 된다. attention 가중치가 반환되지 않아 VLA-Cache의 task 관련 토큰 선택이 동작하지 않는다.
- skip이 연속되면 VLA-Cache의 비교 간격이 길어져 재사용 토큰 수가 줄어든다. 이는 의도된 동작이다.
