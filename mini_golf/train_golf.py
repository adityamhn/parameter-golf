"""
Mini-golf training script for M4 Pro. THIS IS THE ONLY FILE THE AGENT MODIFIES.

Architecture: U-Net skip GPT (same as parameter-golf baseline) with scaled-down defaults.
Optimizer: Muon (matrix params) + Adam (embeddings, scalars).
Backend: MLX on Apple Silicon.

Usage: python3 train_golf.py
"""
from __future__ import annotations

import math
import os
import pickle
import time
import zlib
from pathlib import Path

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

import sentencepiece as spm

from prepare_golf import (
    MAX_SEQ_LEN,
    TIME_BUDGET as _BASE_TIME_BUDGET,
    VOCAB_SIZE,
    EVAL_TOKENS,
    VAL_BATCH_TOKENS,
    COMPUTE_DTYPE,
    DATA_PATH,
    TOKENIZER_PATH,
    CONTROL_TENSOR_NAME_PATTERNS,
    TokenLoader,
    load_validation_tokens,
    build_sentencepiece_luts,
    evaluate_bpb,
    quantize_state_dict_int8,
    dequantize_state_dict_int8,
)

TIME_BUDGET = 600  # Tier 2: match H100 challenge time budget (override prepare_golf's 90s)

# ---------------------------------------------------------------------------
# Hyperparameters (tune these!)
# ---------------------------------------------------------------------------

NUM_LAYERS = 9
MODEL_DIM = 384
NUM_HEADS = 6
NUM_KV_HEADS = 3
MLP_MULT = 2
TRAIN_SEQ_LEN = MAX_SEQ_LEN  # 512
LOGIT_SOFTCAP = 30.0
ROPE_BASE = 10000.0
QK_GAIN_INIT = 1.5
TIED_EMBED_INIT_STD = 0.005

TRAIN_BATCH_TOKENS = 8192
GRAD_ACCUM_STEPS = 1
MAX_ITERATIONS = 50000
WARMUP_STEPS = 5

# Optimizer
TIED_EMBED_LR = 0.05
MATRIX_LR = 0.04
SCALAR_LR = 0.04
MUON_MOMENTUM = 0.95
MUON_BACKEND_STEPS = 5
MUON_MOMENTUM_WARMUP_START = 0.85
MUON_MOMENTUM_WARMUP_STEPS = 200
BETA1 = 0.9
BETA2 = 0.95
ADAM_EPS = 1e-8
WARMDOWN_ITERS = 4000
MUON_WD = 0.04
ADAM_WD = 0.01
SEED = 1337

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def lr_mul(step: int, elapsed_ms: float) -> float:
    if WARMDOWN_ITERS <= 0:
        return 1.0
    step_ms = elapsed_ms / max(step, 1)
    warmdown_ms = WARMDOWN_ITERS * step_ms
    remaining_ms = max(1000.0 * TIME_BUDGET - elapsed_ms, 0.0)
    return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


# ---------------------------------------------------------------------------
# Model blocks
# ---------------------------------------------------------------------------

class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x * x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(
            np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32)))
        )

    def __call__(self, x: mx.array, x0: mx.array) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_softcap = LOGIT_SOFTCAP
        self.tok_emb = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        self.num_encoder_layers = NUM_LAYERS
        self.num_decoder_layers = 0
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, MODEL_DIM), dtype=mx.float32)
        self.blocks = [
            Block(MODEL_DIM, NUM_HEADS, NUM_KV_HEADS, MLP_MULT, ROPE_BASE, QK_GAIN_INIT)
            for _ in range(NUM_LAYERS)
        ]
        self.final_norm = RMSNormNoWeight()

        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * TIED_EMBED_INIT_STD
        ).astype(COMPUTE_DTYPE)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        skips: list[mx.array] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
        logits = self.softcap(logits_proj)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

# ---------------------------------------------------------------------------
# Optimizers (Muon + Adam split)
# ---------------------------------------------------------------------------

class Muon:
    def __init__(self, keys: list[str], params: dict[str, mx.array]):
        self.keys = keys
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array],
             step: int, lr_scale: float) -> dict[str, mx.array]:
        if MUON_MOMENTUM_WARMUP_STEPS:
            t = min(step / MUON_MOMENTUM_WARMUP_STEPS, 1.0)
            momentum = (1.0 - t) * MUON_MOMENTUM_WARMUP_START + t * MUON_MOMENTUM
        else:
            momentum = MUON_MOMENTUM
        lr = MATRIX_LR * lr_scale
        out: dict[str, mx.array] = {}
        for k in self.keys:
            p, g = params[k], grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, MUON_BACKEND_STEPS)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            p_decayed = p * (1.0 - lr * MUON_WD) if MUON_WD > 0 else p
            out[k] = p_decayed - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    def __init__(self, model: GPT):
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k for k, p in params.items()
            if k.startswith("blocks.") and p.ndim == 2
            and not any(pat in k for pat in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k for k, p in params.items()
            if k == "skip_weights"
            or (k.startswith("blocks.") and (p.ndim < 2 or any(pat in k for pat in CONTROL_TENSOR_NAME_PATTERNS)))
        ]
        self.muon = Muon(self.matrix_keys, params)
        self.adam_embed = optim.Adam(
            learning_rate=TIED_EMBED_LR, betas=[BETA1, BETA2], eps=ADAM_EPS, bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=SCALAR_LR, betas=[BETA1, BETA2], eps=ADAM_EPS, bias_correction=True,
        )

    def step(self, model: GPT, grads_tree: dict, step: int, lr_scale: float) -> None:
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)
        updated.update(self.muon.step(params, grads, step=step, lr_scale=lr_scale))
        self.adam_embed.learning_rate = TIED_EMBED_LR * lr_scale
        embed_updated = self.adam_embed.apply_gradients(
            {self.embed_key: grads[self.embed_key]},
            {self.embed_key: params[self.embed_key]},
        )
        if ADAM_WD > 0:
            wd_factor = 1.0 - TIED_EMBED_LR * lr_scale * ADAM_WD
            embed_updated = {k: v * wd_factor for k, v in embed_updated.items()}
        updated.update(embed_updated)
        self.adam_scalar.learning_rate = SCALAR_LR * lr_scale
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))
        model.update(tree_unflatten(list(updated.items())))


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------

def accumulate_flat_grads(
    accum: dict[str, mx.array] | None,
    grads_tree: dict,
    scale: float,
) -> dict[str, mx.array]:
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"mini-golf | layers={NUM_LAYERS} dim={MODEL_DIM} heads={NUM_HEADS} "
          f"kv_heads={NUM_KV_HEADS} mlp_mult={MLP_MULT} seq_len={TRAIN_SEQ_LEN}")
    print(f"batch_tokens={TRAIN_BATCH_TOKENS} grad_accum={GRAD_ACCUM_STEPS} "
          f"time_budget={TIME_BUDGET}s seed={SEED}")

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    assert int(sp.vocab_size()) == VOCAB_SIZE, (
        f"Tokenizer vocab {sp.vocab_size()} != VOCAB_SIZE {VOCAB_SIZE}"
    )
    val_files = os.path.join(DATA_PATH, "fineweb_val_*.bin")
    train_files = os.path.join(DATA_PATH, "fineweb_train_*.bin")
    val_tokens = load_validation_tokens(val_files, TRAIN_SEQ_LEN)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, VOCAB_SIZE)

    mx.random.seed(SEED)
    train_loader = TokenLoader(train_files, log_fn=print)

    model = GPT()
    opt = SplitOptimizers(model)

    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    print(f"model_params:{n_params} ({n_params / 1e6:.2f}M)")

    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state, outputs=model.state,
    )

    # Warmup: prime compile graphs without updating parameters.
    microbatch_tokens = TRAIN_BATCH_TOKENS // GRAD_ACCUM_STEPS
    for _ in range(WARMUP_STEPS):
        warmup_loss = mx.array(0.0, dtype=mx.float32)
        for _ in range(GRAD_ACCUM_STEPS):
            x, y = train_loader.next_batch(microbatch_tokens, TRAIN_SEQ_LEN)
            loss, grads = compiled_loss_and_grad(x, y)
            warmup_loss = warmup_loss + loss
            mx.eval(warmup_loss, grads)
        mx.synchronize()
    print(f"warmup:{WARMUP_STEPS} steps done")

    # Reset loader so training starts from a clean token window.
    train_loader = TokenLoader(train_files, log_fn=print)

    # Training loop
    train_time_ms = 0.0
    t0 = time.perf_counter()
    step = 0
    stop_after_step: int | None = None

    while True:
        last_step = step == MAX_ITERATIONS or (stop_after_step is not None and step >= stop_after_step)
        if last_step:
            break

        lrm = lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()

        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / GRAD_ACCUM_STEPS
        for _ in range(GRAD_ACCUM_STEPS):
            x, y = train_loader.next_batch(microbatch_tokens, TRAIN_SEQ_LEN)
            loss, grads = compiled_loss_and_grad(x, y)
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale
            mx.eval(train_loss, accum)

        grads_tree = tree_unflatten(list(accum.items()))
        train_loss_value = float(train_loss.item())
        opt.step(model, grads_tree, step=step, lr_scale=lrm)
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = TRAIN_BATCH_TOKENS / (step_ms / 1000.0)
        step += 1

        if step <= 5 or step % 100 == 0:
            print(
                f"step:{step}/{MAX_ITERATIONS} loss:{train_loss_value:.4f} "
                f"lr_mul:{lrm:.3f} dt:{step_ms:.0f}ms tok/s:{tok_s:.0f} "
                f"train_time:{approx_train_time_ms / 1000.0:.1f}s"
            )

        if stop_after_step is None and approx_train_time_ms >= 1000.0 * TIME_BUDGET:
            stop_after_step = step

    total_train_time = time.perf_counter() - t0
    print(f"\ntraining done: {step} steps in {total_train_time:.1f}s")

    # Final evaluation
    print("evaluating...")
    val_loss, val_bpb = evaluate_bpb(
        compiled_loss, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        seq_len=TRAIN_SEQ_LEN,
        log_fn=print,
    )

    # Quantize and measure artifact size
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    quant_obj, quant_stats = quantize_state_dict_int8(flat_state)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)

    code_bytes = len(Path(__file__).read_text(encoding="utf-8").encode("utf-8"))
    model_bytes = len(quant_blob)
    artifact_bytes = code_bytes + model_bytes

    # Roundtrip: reload quantized weights and re-evaluate
    quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob)))
    model.update(tree_unflatten(list(quant_flat.items())))
    rt_val_loss, rt_val_bpb = evaluate_bpb(
        compiled_loss, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        seq_len=TRAIN_SEQ_LEN,
        log_fn=print,
    )

    # Summary block (parsed by the agent)
    print("---")
    print(f"val_bpb:          {rt_val_bpb:.6f}")
    print(f"val_loss:         {rt_val_loss:.6f}")
    print(f"pre_quant_bpb:    {val_bpb:.6f}")
    print(f"training_seconds: {total_train_time:.1f}")
    print(f"total_tokens_M:   {step * TRAIN_BATCH_TOKENS / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {n_params / 1e6:.2f}")
    print(f"depth:            {NUM_LAYERS}")
    print(f"artifact_bytes:   {artifact_bytes}")
    print(f"code_bytes:       {code_bytes}")
    print(f"model_bytes:      {model_bytes}")


if __name__ == "__main__":
    main()
