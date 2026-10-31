"""
attn_stats="full"(원래 eager attention map)과 "lite"(SDPA + 프리필 통계)가 같은 재사용 결정을 내리는지 확인한다.

같은 두 프레임으로 두 방식을 각각 2스텝씩 돌린 뒤 비교한다.
  1) 재사용 후보 토큰(reusable_patches)이 같은지
  2) 레이어별 재사용 비율(proportion_attn_var) 차이
  3) 15번 레이어 text→vision 점수 / 레이어별 엔트로피 차이
  4) 두 번째 스텝 action이 같은지

실행 (src/openvla에서):
  python experiments/robot/libero/check_attn_stats_lite.py \
      --checkpoint checkpoints/openvla-7b-finetuned-libero-spatial
  # 실제 프레임 두 장으로 확인하려면 --img1 a.png --img2 b.png
"""

import argparse
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from experiments.robot.openvla_utils import get_processor, get_vla, get_vla_action
from experiments.robot.vla_cache_utils import token_attention_merge


def load_frames(args):
    if args.img1 and args.img2:
        img1 = np.array(Image.open(args.img1).convert("RGB").resize((224, 224)))
        img2 = np.array(Image.open(args.img2).convert("RGB").resize((224, 224)))
        return img1, img2
    # 합성 프레임: 대부분 그대로 두고 일부 영역만 바꿔서 정적 패치가 생기게 한다
    rng = np.random.default_rng(0)
    img1 = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    img2 = img1.copy()
    img2[80:140, 90:150] = rng.integers(0, 256, size=(60, 60, 3), dtype=np.uint8)
    return img1, img2


def run_two_steps(cfg, vla, processor, img1, img2, task):
    obs1 = {"full_image": img1, "prev_image": img1}
    obs2 = {"full_image": img2, "prev_image": img1}
    _, caches1, _ = get_vla_action(cfg, vla, processor, cfg.pretrained_checkpoint, obs1, task,
                                   cfg.unnorm_key, center_crop=cfg.center_crop, last_caches=None)
    keep = {k: v for k, v in caches1.items() if k != "past_key_values"}   # 비교용으로 통계만 보관
    action2, _, _ = get_vla_action(cfg, vla, processor, cfg.pretrained_checkpoint, obs2, task,
                                   cfg.unnorm_key, center_crop=cfg.center_crop, last_caches=caches1)
    lm_cfg = vla.language_model.config
    patches = lm_cfg.reusable_patches
    prop = lm_cfg.proportion_attn_var
    return {
        "first_step": keep,
        "action2": np.asarray(action2),
        "patches": None if patches is None else patches.tolist(),
        "prop": None if prop is None else prop.float().cpu(),
    }


def entropies_from_full(attn):
    ents = []
    for a in attn[:-1]:
        a = a.mean(dim=1)[0]
        a /= a.sum(dim=-1, keepdim=True) + 1e-10
        a = torch.nan_to_num(a, nan=0.0)
        ents.append((-torch.sum(a * torch.log(a + 1e-10), dim=-1)).mean())
    return torch.stack(ents).float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--unnorm_key", default="libero_spatial")
    ap.add_argument("--task", default="pick up the black bowl between the plate and the ramekin and place it on the plate")
    ap.add_argument("--img1", default=None)
    ap.add_argument("--img2", default=None)
    args = ap.parse_args()

    base = dict(pretrained_checkpoint=args.checkpoint, load_in_8bit=False, load_in_4bit=False,
                use_vla_cache=True, verbose=False, unnorm_key=args.unnorm_key, center_crop=True)
    cfg_full = SimpleNamespace(**base, attn_stats="full")
    cfg_lite = SimpleNamespace(**base, attn_stats="lite")

    vla = get_vla(cfg_full)
    processor = get_processor(cfg_full)
    print("attn_implementation:", vla.language_model.config._attn_implementation)
    img1, img2 = load_frames(args)

    with torch.no_grad():
        full = run_two_steps(cfg_full, vla, processor, img1, img2, args.task)
        torch.cuda.empty_cache()
        lite = run_two_steps(cfg_lite, vla, processor, img1, img2, args.task)

    # 3) 통계 비교 (첫 스텝 프리필)
    attn = full["first_step"]["attentions"]
    stats = lite["first_step"]["attn_stats"]
    tv_full = token_attention_merge(attn).float()
    tv_lite = stats["tv_score"].float()
    ent_full = entropies_from_full(attn)
    ent_lite = stats["entropies"].float().cpu()
    print(f"[tv_score]   max |diff| = {(tv_full - tv_lite).abs().max().item():.3e}")
    print(f"[entropies]  max |diff| = {(ent_full - ent_lite).abs().max().item():.3e}")

    # 1) 2) 재사용 결정 비교 (두 번째 스텝)
    same_patches = full["patches"] == lite["patches"]
    n = 0 if full["patches"] is None else len(full["patches"])
    print(f"[reusable_patches] same = {same_patches}  (full: {n}개)")
    if full["prop"] is not None and lite["prop"] is not None:
        print(f"[proportion] max |diff| = {(full['prop'] - lite['prop']).abs().max().item():.3e}")
    else:
        print(f"[proportion] full={full['prop'] is not None}, lite={lite['prop'] is not None}")

    # 4) action 비교
    same_action = np.array_equal(full["action2"], lite["action2"])
    print(f"[action step2] same = {same_action}")
    if not same_action:
        print("  full:", full["action2"])
        print("  lite:", lite["action2"])
        print("  (SDPA와 eager는 bf16 누적 순서가 달라 디코드에서 토큰이 가끔 달라질 수 있다. "
              "재사용 후보·비율이 같으면 로직은 같은 것)")


if __name__ == "__main__":
    main()