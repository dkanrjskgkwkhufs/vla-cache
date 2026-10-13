import ast, copy, types, numpy as np, sys
"""
SP-VLA 원본 skip 로직과 run_libero_eval.py의 ActionScheduler가 같은 결정을 내리는지 비교.
GPU·모델 없이 numpy만으로 실행된다.
  python check_skip_equivalence.py <SP-VLA/prismatic/extern/hf/modeling_prismatic.py> <run_libero_eval.py>
"""
SP = sys.argv[1]
src = open(SP).read(); lines = src.splitlines()
tree = ast.parse(src)
cls = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "OpenVLAForActionPrediction"][0]
meths = {f.name: ast.get_source_segment(src, f) for f in cls.body if isinstance(f, ast.FunctionDef)}

# SP-VLA 원본 메서드 그대로 (들여쓰기 제거 후 exec)
import textwrap
ns = {"np": np, "copy": copy}
for m in ["compressor_init", "fit_next_action", "save_actions"]:
    exec(textwrap.dedent(meths[m]), ns)

# predict_action의 skip 판단 구간(원본 822~854행)을 그대로 잘라서 사용
pa = meths["predict_action"]
start = pa.index("# Step 1: Calculate translation magnitude")
end = pa.index("else:\n            # ===== INFERENCE PATH")
skip_block = textwrap.dedent("        " + pa[start:end].strip("\n").lstrip())
class SP:
    pass
for m in ["compressor_init", "fit_next_action", "save_actions"]:
    setattr(SP, m, ns[m])
SP.save_action_number = 6
def sp_step(self, model_action):
    """원본 skip 블록 실행. skip이면 회귀 action, 아니면 model_action을 '생성'으로 저장."""
    self.cfg.skip_flag = 0   # 원본 else 분기 첫 줄과 동일
    loc = {"self": self, "np": np}
    exec(skip_block, {"np": np}, loc)
    if self.cfg.skip_flag == 1:
        return loc["actions"], True
    self.save_actions(model_action, generate=True)
    return model_action, False
SP.step = sp_step

ours_src = open(sys.argv[2]).read()
t2 = ast.parse(ours_src)
AS = [n for n in t2.body if isinstance(n, ast.ClassDef) and n.name == "ActionScheduler"][0]
ns2 = {"np": np}; exec(ast.get_source_segment(ours_src, AS), ns2)
ActionScheduler = ns2["ActionScheduler"]

def run(trial, n_steps, step_skip, zxy, zmax, rng):
    sp = SP(); sp.compressor_init()
    sp.cfg = types.SimpleNamespace(step_skip=step_skip, z_xy_rate_skip=zxy, z_max_skip=zmax, single_move=[], task_skip=0, skip_flag=0)
    ours = ActionScheduler(zxy, zmax, step_skip, 6, 4/6)
    # 실제 궤적 비슷한 action: 평면 이동 구간과 수직 이동 구간이 섞인 시퀀스
    mismatch = 0
    ours_skips = 0
    for k in range(n_steps):
        phase = (k // 15) % 3
        base = np.array([0.4, 0.2, 0.02]) if phase == 0 else (np.array([0.05, 0.02, -0.5]) if phase == 1 else np.array([-0.3, 0.1, 0.05]))
        model_action = np.concatenate([base + rng.normal(0, 0.08, 3), rng.normal(0, 0.05, 3), [float(rng.random() > 0.5)]])
        a_sp, sk_sp = sp.step(model_action.copy())
        if ours.should_skip():
            a_o, sk_o = ours.predict(), True
            ours.record(a_o, generated=False)
            ours_skips += 1
        else:
            a_o, sk_o = model_action.copy(), False
            ours.record(a_o, generated=True)
        if sk_sp != sk_o or not np.allclose(a_sp, a_o, atol=1e-12):
            mismatch += 1
    return mismatch, sp.total_skip_step, ours_skips

rng = np.random.default_rng(0)
tot = 0
for params in [(1, 0.4, 0.3), (2, 0.6, 0.5), (2, 1.2, 0.3), (2, 1.0, 0.3)]:
    for trial in range(200):
        m, s1, s2 = run(trial, 220, *params, rng)
        tot += m
        assert s1 == s2, (params, trial, s1, s2)
    print("params", params, "OK, last episode skips:", s1, s2)
print("total mismatched steps:", tot)