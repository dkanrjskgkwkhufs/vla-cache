"""
여러 조건의 *_summary.json 을 모아 KV 캐시 실험 문서와 같은 형식의 표로 출력한다.
기준 조건(--base, 기본 C0) 대비 speedup / FLOPs(%)도 계산한다 (SP-VLA 논문 Table 1 방식).

  python compare_runs.py experiments/logs/*_summary.json
  python compare_runs.py experiments/logs/*_summary.json --base B1
"""

import argparse
import json


COLS = [
    ("cond", "방법", "{}"),
    ("success", "성공률", "{}"),
    ("skip_rate", "skip률", "{:.1%}"),
    ("t_step_ms_median", "스텝 전체(ms)", "{:.1f}"),
    ("t_select_ms_median", "select", "{:.1f}"),
    ("t_vision_ms_median", "비전", "{:.1f}"),
    ("t_prefill_ms_median", "프리필", "{:.1f}"),
    ("t_decode_ms_median", "디코드", "{:.1f}"),
    ("control_hz", "제어 Hz", "{:.2f}"),
    ("effective_hz", "실효 Hz(skip 포함)", "{:.2f}"),
    ("speedup", "speedup", "{:.2f}x"),
    ("flops_pct", "FLOPs(%)", "{:.1f}"),
    ("reuse", "재사용 토큰", "{}"),
]


def fmt(v, f):
    if v is None:
        return "-"
    try:
        return f.format(v)
    except (ValueError, TypeError):
        return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--base", default="C0", help="speedup / FLOPs 기준 조건 (C0, B1, ...)")
    args = ap.parse_args()

    runs = [json.load(open(p)) for p in args.summaries]
    base = next((r for r in runs if r.get("cond") == args.base), None)
    if base is None:
        print(f"[warn] 기준 조건 {args.base} 결과가 없어 speedup / FLOPs(%)는 비워 둡니다.")

    rows = []
    for r in runs:
        row = dict(r)
        row["success"] = f'{r["success_rate"] * 100:.1f}% ({r["successes"]}/{r["episodes"]})'
        row["reuse"] = (f'{r["reuse_cand_median"]:.0f} ({r["reuse_cand_min"]}~{r["reuse_cand_max"]})'
                        if r.get("reuse_cand_median") is not None else None)
        if base and base.get("effective_hz") and r.get("effective_hz"):
            row["speedup"] = r["effective_hz"] / base["effective_hz"]
        if base and base.get("prefill_tflops_per_step") and r.get("prefill_tflops_per_step") is not None:
            row["flops_pct"] = 100 * r["prefill_tflops_per_step"] / base["prefill_tflops_per_step"]
        rows.append(row)

    print("| " + " | ".join(c[1] for c in COLS) + " |")
    print("|" + "---|" * len(COLS))
    for row in rows:
        print("| " + " | ".join(fmt(row.get(k), f) for k, _, f in COLS) + " |")
    print(f"\n- 시간: 모델 호출 스텝 중앙값, 에피소드별 첫 {runs[0].get('timing_excluded_first_steps')}스텝 제외")
    print(f"- speedup: 실효 Hz(skip 포함 평균) / {args.base}의 실효 Hz")
    print(f"- FLOPs(%): 스텝당 언어모델 프리필 연산량 / {args.base} 값")


if __name__ == "__main__":
    main()