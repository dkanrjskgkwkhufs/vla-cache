# [thor] 평가에는 prismatic.extern.hf 만 필요하다.
# 학습용 코드(prismatic.models → vla.datasets → dlimp / tensorflow)는 Thor에 없으므로, 없으면 건너뛴다.
try:
    from .models import available_model_names, available_models, get_model_description, load
except ImportError as e:  # dlimp, tensorflow 등 학습 전용 의존성 없음
    import warnings

    warnings.warn(f"[thor] prismatic training modules not loaded ({e}). Eval-only mode.")