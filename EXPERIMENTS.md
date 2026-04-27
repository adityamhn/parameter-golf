# Mini-Golf Autoresearch Experiments — Branch & Tier Summary

This document collates every experiment performed across the `mini-golf/*` autoresearch branches in this repo.
The autoresearch loop runs locally on a single RTX 3060 12GB GPU using `train_gpt_single_gpu.py` as a proxy
for the full-scale 8×H100 challenge described in [README.md](README.md).

The experiments split into two **tiers**:

- **Tier 1 (proxy)** — small-budget, fast-iteration runs. ~700–2000 steps, batch ~98k tokens/step.
  Used to discover ideas. Most branches operated here.
- **Tier 2 (scaled proxy)** — wider/deeper model (9L 384D, U-Net) and 600 s wall-clock budget,
  matched to full-scale H100 ratios. Used to gate Tier 1 wins before promoting.

The `keep` / `discard` status reflects whether the run beat the running local best (a `keep` is a *local-best*
under that branch's proxy, NOT a leaderboard submission).

---

## Branch order (chronological)

| Branch | Focus | Best local val_bpb |
|--------|-------|--------------------:|
| [`mini-golf/mar27`](#mini-golfmar27)         | Initial setup + activation/depth bumps   | (see Tier 2 results) |
| [`mini-golf/mar27b`](#mini-golfmar27b)       | More depth + EMA                         | (see Tier 2 results) |
| [`mini-golf/mar27c`](#mini-golfmar27c)       | Tier 1: WD + U-Net asymmetry             | (see Tier 2 results) |
| [`mini-golf/mar27c-t2`](#mini-golfmar27c-t2) | Tier 2: warmdown sweep + plain xfmr      | **1.6513** (warmdown 3500) |
| [`mini-golf/mar30`](#mini-golfmar30)         | LEAKY/RoPE sweeps → 12L Tier1 SOTA       | **1.3354** (40a50ed, 12L KV2 ROPE23 LEAKY0.72) |
| [`mini-golf/apr1-tier1-loop`](#mini-golfapr1-tier1-loop) | Autoresearch loop on Tier-1 best stack | 1.329 (multi-knob combos) |
| [`mini-golf/apr1-newtop-loop`](#mini-golfapr1-newtop-loop) | XSA-all + bigram + Muon-momentum sweep | **1.3294** (MUON_MOMENTUM_WARMUP_START=0.92) |

---

## `mini-golf/mar27`

First proxy branch. Walks from the unmodified `train_golf.py` baseline up through small known-good wins.

| Commit  | Experiment | Notes |
|---------|------------|-------|
| `fa32f92` | **Baseline:** initial mini-golf setup | Unmodified `train_golf.py`, 4-layer 256-dim U-Net skip GPT |
| `8edbee7` | LeakyReLU(0.5)² instead of ReLU² | Preserves negative-grad flow, eliminates dead neurons. Proven at full scale (~−0.003 BPB) |
| `e4b279e` | Depth 4 → 5 layers | Depth more parameter-efficient than width |

This branch laid down the chassis used by every later branch.

---

## `mini-golf/mar27b`

Extends `mar27` with two more known-good moves.

| Commit  | Experiment | Notes |
|---------|------------|-------|
| `41b550c` | Depth 5 → 6 layers | Continue the depth ladder |
| `64cf68f` | EMA weight averaging (decay 0.997) | Standard nanoGPT-style smoothing |

---

## `mini-golf/mar27c` (Tier 1)

Focus: weight decay + U-Net asymmetry. Six experiments, each commit a single change.

| Commit  | Experiment | Notes |
|---------|------------|-------|
| `8839d20` | **Baseline:** 6L 256D LeakyReLU(0.5)² with EMA removed | EMA's 333-step window averaged in poor early weights at 1211-step proxy |
| `b254e97` | Muon weight decay WD=0.04 | L2 regularization on matrix params; zero compute overhead |
| `2365e34` | Adam weight decay on embedding (WD=0.01) | Extends regularization to embedding |
| `bf61d68` | Asymmetric U-Net 4 enc + 2 dec | Shift mass toward encoder |
| `2ab7bae` | Asymmetric U-Net 5 enc + 1 dec | Even more encoder-heavy |
| `4335a17` | **Plain transformer 6+0** (no decoder/skips) | Encoder-only limit |

The Tier 1 best stack from this branch graduated to Tier 2.

---

## `mini-golf/mar27c-t2` (Tier 2 — fully gated)

Scaled proxy: 9L / 384D / 6H / 3KV U-Net 4+5, 600 s wall-clock to match full-scale H100 ratios.
Every row below was logged to `mini_golf/results.tsv` with explicit `keep`/`discard`.

**Baseline (`9a299c0`):** 9L 384D U-Net 4+5 with WD — **val_bpb 1.7517** (2832 steps).

### Architecture experiments

| Commit  | Experiment | val_bpb | Status | Notes |
|---------|------------|--------:|--------|-------|
| `4393e26` | 3× MLP width (hidden=1152) | 1.8029 | discard | Slows training (2356 steps, +0.051) |
| `6d9959c` | EMA decay=0.997, start=2000 | 1.9152 | discard | Bad avg window at this step count (+0.164) |
| `11007e7` | Partial RoPE 16/64 | 1.7747 | discard | Concat overhead hurts (+0.023) |
| `c2d2a2a` | 11 layers (dim=384) | 1.7622 | discard | Near-miss; too slow (2306 steps) |
| `32c4e31` | 10 layers (dim=384) | 1.7620 | discard | Still worse (2634 steps) |
| `f60e705` | BigramHash 2048 aux input | 1.7702 | discard | Doesn't help at Tier 2 (+0.019) |
| `64547ca` | **Encoder-heavy U-Net 6+3** | **1.7296** | **keep** | WINNER — −0.022 BPB; 2907 steps |
| `ecfdc4e` | Encoder-heavy U-Net 7+2 | 1.7315 | discard | Slightly worse than 6+3 |
| `cc3a2f3` | **Plain transformer 9+0** | **1.7246** | **keep** | New best — −0.005 from 6+3 |
| `fed3f0a` | Disable logit softcap | 1.6797 | discard | softcap=30 still helps |
| `ef44cf3` | Remove resid_mix on plain xfmr | 1.6745 | discard | x0 blending still needed |

### Optimizer / regularization experiments

| Commit  | Experiment | val_bpb | Status | Notes |
|---------|------------|--------:|--------|-------|
| `ca2501a` | Higher Muon WD=0.08 | 1.7519 | discard | Too aggressive |
| `cfc8362` | grad_accum=2 (effective batch 16k) | 1.7303 | discard | Fewer updates hurts |
| `369e186` | Higher Muon LR=0.06 | 1.7651 | discard | Too aggressive |
| `649f4cc` | Adam WD=0.02 on embedding | 1.6611 | discard | 0.01 is better |
| `d9c3367` | Lower Muon WD=0.02 | 1.6649 | discard | 0.04 is better |
| `7257d77` | QK gain init 2.0 (from 1.5) | 1.6693 | discard | Worse |
| `9f3ae84` | 8H/4KV (head_dim=48, matches H100) | 1.6759 | discard | Slower per step |

### Warmdown sweep (the big win on this branch)

Linear warmdown length, in iters:

| Commit  | Warmdown iters | val_bpb | Status | Cumulative Δ from baseline |
|---------|---------------:|--------:|--------|---------------------------:|
| `77b9269` |   600 | 1.7111 | keep | −0.041 |
| `d72b4b0` |   800 | 1.6934 | keep | −0.058 |
| `3beebad` |  1000 | 1.6873 | keep | −0.064 |
| `65fcaf8` |  1200 | 1.6810 | keep | −0.071 |
| `7e0dd0b` |  1500 | 1.6777 | keep | −0.074 |
| `8ab73bc` |  2000 | 1.6671 | keep | −0.085 |
| `695e8bb` |  2500 | 1.6641 | keep | −0.088 |
| `e442182` |  3000 | 1.6614 | keep | −0.090 |
| `9ae7695` | **3500** | **1.6513** | **keep** | **−0.101 (best)** |
| `7dbeb83` |  4000 | 1.6691 | discard | Initial LR too low |
| `a32e698` | 50000 (full decay) | 1.7041 | discard | Initial LR way too low |
| `8a3b774` | cosine warmdown | 1.6629 | discard | Linear is better |

**Tier 2 outcome:** plain transformer 9+0 + warmdown 3500 → **val_bpb 1.6513**. These findings were applied to
`train_gpt.py` in commit `caccc9e`.

---

## `mini-golf/mar30`

Tier 1 sweeps starting from the SOTA-baseline (`50390d6`, val_bpb 1.4589 on the proxy). Mostly ROPE / LEAKY tuning,
ending in the new **12-layer Tier-1 SOTA** that was promoted to Tier 2.

| Commit  | Experiment | val_bpb | Status | Notes |
|---------|------------|--------:|--------|-------|
| `50390d6` | **SOTA baseline** | 1.4589 | keep | Stopped after proxy metric & size signal |
| `9c4598a` | LEAKY_SLOPE sweep (env'd) | — | — | Wires LEAKY env knob |
| `7cb084a` | LEAKY_SLOPE=0.3 | 1.4645 | discard | Worse than baseline (+0.006) |
| `069183e` | LEAKY_SLOPE=0.7 | (kept) | keep | New direction for slope |
| `a8e70cc` | ROPE_DIMS=18 (LEAKY 0.7) | — | — | First partial-RoPE proxy |
| `9af5197` | ROPE_DIMS=20 (LEAKY 0.7) | — | — | |
| `7a27fd3` | ROPE_DIMS=22 (LEAKY 0.7, EMA wired) | — | — | EMA env honored via `deab3f5` |
| `552ff5d` | ROPE_DIMS=26 ridge (LEAKY 0.7) | — | — | Ridge winner from this sweep |
| `7df6489` | LEAKY_SLOPE=0.72 fine on ROPE26 | — | — | Even finer slope |
| `7d406bf`/`14b9081` | (fix) odd ROPE_DIMS via even rotary prefix | — | — | Enables odd ROPE_DIMS |
| `650d8b1` | ROPE23 + LEAKY 0.72 (odd RoPE) | — | — | Odd-RoPE proxy win |
| `9bbb9b2` | **Tier 1: NUM_LAYERS=12 (ROPE23, LEAKY 0.72)** | — | **keep** | Promoted as Tier-1 SOTA |
| `2bdaf1a` | docs: Tier 1/2 gate policy | — | — | Codifies gating |
| `40a50ed` | **Tier 2: NUM_KV_HEADS=2 (12L ROPE23 LEAKY 0.72)** | **1.3354** | **keep** | Best stack — over 16 MB limit (16.13 MB) |

**Tier 1 reference (recorded in apr1 branch):** `tier1-sota` 11L KV4 ROPE16 LEAKY0.5 → val_bpb **1.3361** (2000 steps).

---

## `mini-golf/apr1-tier1-loop`

Sets `train_gpt_single_gpu.py` defaults to the `40a50ed` best stack (12L KV2 ROPE23 LEAKY0.72) and runs
the **autoresearch infinite loop** as defined in `mini_golf/program.md` (rewritten in `0cc6dcb`). Each row in
`mini_golf/results.tsv` is one full proxy run logged by the loop. Most rows share the `0cc6dcb` commit because
the loop modified env vars, not the file content.

Reference rows:

| Tag/Commit | val_bpb | mem | size | Status | Description |
|------------|--------:|---:|----:|--------|-------------|
| `tier1-sota` | 1.3361 | 3.3 | 13.68 MB | keep | SOTA baseline 11L KV4 ROPE16 LEAKY0.5 (2000 steps reference) |
| `40a50ed`    | 1.3354 | 3.4 | 16.13 MB | keep | best stack 12L KV2 ROPE23 LEAKY0.72 (over 16 MB!) |

### Single-knob sweeps on the 12L stack (all commit `0cc6dcb`)

Attention / KV / RoPE:

| Knob | Value | val_bpb | Status |
|------|-------|--------:|--------|
| NUM_KV_HEADS | 1 (MQA) | 1.3392 | discard |
| ROPE_DIMS | 20 | 1.3322 | discard |
| ROPE_DIMS | 24 | 1.3362 | discard |
| ROPE_DIMS | 26 | 1.3351 | discard |
| ROPE_DIMS | 32 | 1.3348 | discard |
| ROPE_BASE | 50000  | 1.3317 | discard |
| ROPE_BASE | 100000 | 1.3323 | discard |
| XSA_LAST_N | 3 | 1.3321 | discard |
| XSA_LAST_N | 6 | 1.3349 | discard |
| XSA_LAST_N | 8 | 1.3379 | discard |

Activation / capacity:

| Knob | Value | val_bpb | Status |
|------|-------|--------:|--------|
| LEAKY_SLOPE | 0.60 | 1.3367 | discard |
| LEAKY_SLOPE | 0.65 | 1.3320 | discard |
| LEAKY_SLOPE | 0.80 | 1.3316 | discard |
| MLP_MULT    | 2.5 | 1.3403 | discard |
| MLP_MULT    | 3.5 | 1.3341 | discard |
| MLP_MULT    | 4.0 | 1.3345 | discard |
| NUM_LAYERS  | 10 | 1.3441 | discard |
| NUM_LAYERS  | 11 | 1.3382 | discard |
| NUM_LAYERS  | 13 | 1.3310 | discard |

Optimizer / regularization:

| Knob | Value | val_bpb | Status |
|------|-------|--------:|--------|
| MUON_WD = ADAM_WD | 0.01 | 1.3290 | discard (best raw, but +size) |
| MUON_WD = ADAM_WD | 0.02 | 1.3299 | discard |
| MUON_WD = ADAM_WD | 0.06 | 1.3365 | discard |
| GRAD_CLIP_NORM | 0.2 | 1.3351 | discard |
| GRAD_CLIP_NORM | 0.4 | 1.3357 | discard |
| MUON_MOMENTUM_WARMUP_START | 0.85 | 1.3354 / 1.3358 / 1.3359 (3 reps) | **keep** |
| MUON_MOMENTUM_WARMUP_START | 0.88 | 1.3335 / 1.3334 / 1.3339 (3 reps) | **keep** |

### Combination sweeps (all `0cc6dcb`)

| Combination | val_bpb | Status |
|-------------|--------:|--------|
| ROPE_BASE=50000 + NUM_KV_HEADS=1 | 1.3392 / 1.3398 / 1.3390 | discard |
| ROPE_BASE=50000 + NUM_LAYERS=11 | 1.3355 / 1.3358 / 1.3364 | keep (×2 / discard) |
| ROPE_BASE=50000 + MLP_MULT=2.5 | 1.3412 / 1.3415 / 1.3408 | discard |
| LEAKY_SLOPE=0.80 + NUM_KV_HEADS=1 | 1.3402 / 1.3407 / 1.3412 | discard |
| XSA_LAST_N=3 + NUM_KV_HEADS=1 | 1.3414 / 1.3418 / 1.3422 | discard |
| NUM_KV_HEADS=1 + ROPE_DIMS=24 | 1.3382 / 1.3402 / 1.3381 | discard |
| ROPE_DIMS=24 + MUON_WD=0.02 + ADAM_WD=0.02 | 1.3339 / 1.3342 / 1.3342 | discard |
| ROPE_BASE=100000 + ROPE_DIMS=24 | 1.3328 / 1.3326 / 1.3335 | discard |
| GRAD_CLIP_NORM=0.2 + MUON_WD=0.02 + ADAM_WD=0.02 | 1.3295 / 1.3297 / 1.3327 | discard (size) |

The clear pattern: nothing single-knob beat the `40a50ed` stack by more than noise except higher Muon/Adam WD,
which won on val_bpb but pushed the artifact over 16 MB. The `MUON_MOMENTUM_WARMUP_START` lever was the most
robust safe win, which set up the `apr1-newtop-loop` follow-ups.

---

## `mini-golf/apr1-newtop-loop`

Continuation loop. Pivots from the Tier-1 stack toward **XSA-all + larger bigram hash + tuned Muon momentum warmup**.

| Commit | Experiment | val_bpb | mem | size | Status |
|--------|------------|--------:|----:|-----:|--------|
| `76a849a` | **Default to XSA-all** on MUON_MOMENTUM_WARMUP_START=0.88 stack | 1.3320 | 3.5 | 15.88 MB | **keep** |
| `de3cdb3` | + BIGRAM_VOCAB_SIZE=3072, BIGRAM_DIM=112 | 1.3320 | 3.5 | 15.98 MB | discard |
| `5125010` | + BIGRAM_VOCAB_SIZE=3072, BIGRAM_DIM=128 | 1.3323 | 3.5 | 16.06 MB | discard |
| `b387530` | + BIGRAM_VOCAB_SIZE=1536, BIGRAM_DIM=112 | 1.3309 | 3.5 | 15.03 MB | **keep** |
| `a45bf8d` | MUON_MOMENTUM_WARMUP_START=0.87 on XSA-all bigram stack | 1.3317 | 3.5 | 14.93 MB | discard |
| `c690c08` | MUON_MOMENTUM_WARMUP_START=0.89 on XSA-all bigram stack | 1.3305 | 3.5 | 15.14 MB | **keep** |
| `b39d16c` | XSA-all + MUON 0.89 + BIGRAM_DIM=120 | 1.3359 | 3.5 | 15.01 MB | discard |
| `8e28026` | MUON_MOMENTUM_WARMUP_START=0.90 on XSA-all bigram stack | 1.3299 | 3.5 | 15.16 MB | **keep** |
| `e34451a` | MUON_MOMENTUM_WARMUP_START=0.91 on XSA-all bigram stack | 1.3295 | 3.5 | 15.24 MB | **keep** |
| `01e036e` | **MUON_MOMENTUM_WARMUP_START=0.92** on XSA-all bigram stack | **1.3294** | 3.5 | 16.14 MB | **keep (current best)** |

**Net trajectory on this branch:** 1.3354 (40a50ed) → 1.3294 (`01e036e`), a −0.006 BPB local improvement,
holding at the 16 MB boundary.

---

## Summary of cross-tier movement

```
Tier 2 baseline (mar27c-t2)                   1.7517   9L/384D U-Net 4+5
Tier 2 best (mar27c-t2 warmdown 3500)         1.6513   plain xfmr 9+0
Tier 1 SOTA reference                         1.3361   11L KV4 ROPE16 LEAKY0.5
Tier 1 best stack (mar30 → 40a50ed)           1.3354   12L KV2 ROPE23 LEAKY0.72
Apr1-newtop-loop best (01e036e)               1.3294   + XSA-all + bigram 1536×112 + MUON 0.92
```

`tier1-sota` and `40a50ed` are the two reference points used by the autoresearch loops. Every later experiment
is measured as a delta against those. Where a row improves val_bpb but pushes the artifact past the 16 MB cap,
the autoresearch policy is to *keep as research champion* and try to recover bytes in follow-up experiments —
that is why several `keep` rows above show artifact sizes >16 MB.

---

## How to read this

- **`keep`** = local best at the moment of the run.
- **`discard`** = strictly worse than the running best (or violates a hard constraint with no offsetting win).
- **val_bpb** numbers across branches are **not directly comparable** because the proxy step count, batch, and
  baseline stack changed between branches. Use the per-branch baseline rows as the calibration anchor.
- **Artifact size** (16 MB cap) is the headline constraint inherited from the parent challenge. Some `keep` rows
  on `apr1-tier1-loop` and `apr1-newtop-loop` are over the cap and need to be slimmed before any leaderboard run.

For the official leaderboard (full 8×H100 runs), see [README.md](README.md).
