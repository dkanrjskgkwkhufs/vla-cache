"""
VLA-Cache용 attention 통계 기록기 (output_attentions 없이 동작)

[원래 방식]
  generate(output_attentions=True)
   → LlamaSdpaAttention이 eager로 바뀜 (프리필 1번 + 디코드 6번, 7번 모두)
   → 32개 레이어의 [1, 32, N, N] attention을 전부 반환하고 다음 스텝까지 들고 감
  그런데 다음 스텝에서 실제로 쓰는 값은 두 가지뿐이다.
   - get_layer_mask_schedule : 레이어별 엔트로피 스칼라 (32개)
   - token_attention_merge   : 15번 레이어 text→vision 점수 (256개)

[바꾼 방식]  transformers 포크 파일은 건드리지 않고 LlamaSdpaAttention.forward만 교체한다.
  - 프리필(q_len > 1) + 기록 켜짐 : attention 확률을 한 번 계산해서 출력도 그걸로 만든다 (중복 계산 없음).
                                   필요한 통계만 남기고 행렬은 바로 버린다.
  - 디코드, 또는 기록 꺼짐        : SDPA
  - output_attentions=True (B1, shadow_compare) : 원래 eager 경로 그대로

[주의] VLA-Cache로 토큰을 뺀 프리필은 q 길이 < k 길이다.
  SDPA의 is_causal=True는 마스크를 왼쪽 위 기준으로 맞추기 때문에 이 경우 결과가 틀린다.
  그래서 cache_position으로 causal 마스크를 직접 만든다 (원래 eager 경로와 같은 마스크).
"""

import math

import torch
import torch.nn.functional as F
import transformers.models.llama.modeling_llama as ml

# vla_cache_utils.token_attention_merge와 같은 위치 정의
V_START = 1
V_END = V_START + 256
T_START = V_END
T_END = T_START + 35
TV_LAYER = 15


class AttnStatsRecorder:
    def __init__(self):
        self.active = False
        self.entropies = {}
        self.tv_score = None

    def start(self):
        self.active = True
        self.entropies = {}
        self.tv_score = None

    def stop(self):
        """기록을 끄고 {'entropies': [L], 'tv_score': [256]}를 돌려준다. 기록된 게 없으면 None."""
        self.active = False
        if not self.entropies or self.tv_score is None:
            return None
        n = max(self.entropies) + 1
        if len(self.entropies) != n:
            return None
        ent = torch.stack([self.entropies[i] for i in range(n)])
        return {"entropies": ent, "tv_score": self.tv_score.cpu()}

    @torch.no_grad()
    def record(self, layer_idx, attn_weights, cache_position):
        # attn_weights: [1, H, q, k], 원래 eager와 같은 dtype (softmax fp32 → query dtype)

        # get_layer_mask_schedule 안의 연산과 동일
        a = attn_weights.mean(dim=1)[0]
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-10)
        a = torch.nan_to_num(a, nan=0.0)
        token_entropy = -torch.sum(a * torch.log(a + 1e-10), dim=-1)
        self.entropies[layer_idx] = token_entropy.mean()

        # token_attention_merge 안의 연산과 동일
        # (원래 코드의 attention_pos = 마지막 cache_position = 이 레이어에 들어온 cache_position)
        if layer_idx == TV_LAYER:
            m = attn_weights.to(torch.float32).squeeze(0).mean(dim=0)
            text_mask = (cache_position >= T_START) & (cache_position < T_END)
            self.tv_score = m[text_mask, V_START:V_END].mean(dim=0)


RECORDER = AttnStatsRecorder()


def _causal_mask_from_positions(cache_position, k_len, dtype, device):
    """query i는 key j <= cache_position[i]만 본다. 모양 [1, 1, q, k]."""
    min_dtype = torch.finfo(dtype).min
    future = torch.arange(k_len, device=device)[None, :] > cache_position[:, None]
    mask = torch.zeros(cache_position.shape[0], k_len, dtype=dtype, device=device)
    mask.masked_fill_(future, min_dtype)
    return mask[None, None]


def _patched_sdpa_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    **kwargs,
):
    # B1 / shadow_compare: 원래 eager 경로 (attention map 반환)
    if output_attentions:
        return ml.LlamaAttention.forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    bsz, q_len, _ = hidden_states.size()

    q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(v, position_ids)
    q, k = ml.apply_rotary_pos_emb(q, k, cos, sin)

    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

    k = ml.repeat_kv(k, self.num_key_value_groups)
    v = ml.repeat_kv(v, self.num_key_value_groups)
    k_len = k.shape[-2]

    if cache_position is None:
        cache_position = torch.arange(k_len - q_len, k_len, device=q.device)

    mask = attention_mask[:, :, :, :k_len] if attention_mask is not None else None
    record = RECORDER.active and q_len > 1

    # 토큰을 뺀 프리필(q_len != k_len)이거나 기록할 때는 마스크를 직접 만든다
    if mask is None and q_len > 1 and (q_len != k_len or record):
        mask = _causal_mask_from_positions(cache_position, k_len, q.dtype, q.device)

    if record:
        # 프리필: 확률을 한 번 계산해서 출력과 통계에 같이 쓴다 (원래 eager와 같은 연산 순서)
        w = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            w = w + mask
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(w, v)
        RECORDER.record(self.layer_idx, w, cache_position)
        del w
    else:
        if mask is not None and q.device.type == "cuda":
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=mask is None and q_len > 1,
        )

    out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
    out = self.o_proj(out)
    return out, None, past_key_value


_INSTALLED = False


def install_attn_stats_patch():
    """LlamaSdpaAttention.forward를 교체한다. 여러 번 불러도 한 번만 적용된다."""
    global _INSTALLED
    if _INSTALLED:
        return
    ml.LlamaSdpaAttention.forward = _patched_sdpa_forward
    _INSTALLED = True