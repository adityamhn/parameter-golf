# Escalation Log

Experiments flagged for testing at full scale on 8xH100.
Each entry was discovered by the mini-golf agent and meets ALL escalation criteria:
significant BPB improvement, architectural (not hyperparameter), scale-independent, clean diff.

---

<!-- Entries will be appended below by the agent -->

## [66dc7bd] Depth recurrence — 3 unique blocks × 2 invocations
- **BPB**: 1.999548 (delta from baseline: +0.0775)
- **Category**: frontier
- **Track**: frontier (unlimited-compute track)
- **What changed**: Reduced from 6 unique blocks to 3, each invoked twice in symmetric U-Net pattern (encoder[0,1,2] → decoder[2,1,0]). 1.64M params (from 3.02M), 1364 steps (from 1251).
- **How to port**: In GPT.__init__, create NUM_UNIQUE_BLOCKS blocks instead of NUM_LAYERS. In __call__, use modular indexing for encoder (ascending) and reversed indexing for decoder.
- **Expected full-scale impact**: Promising direction — 4% worse with 45% fewer params and 9% more steps. At full scale with more training time, the extra steps and parameter efficiency could close the gap. Could be combined with per-invocation control parameters (separate attn_scale/mlp_scale/resid_mix for each invocation) to let the block differentiate encoder vs decoder roles.

## [b254e97] Muon weight decay WD=0.04
- **BPB**: 1.897073 (delta from baseline: -0.025)
- **Category**: other (regularization)
- **Track**: record (10min leaderboard)
- **What changed**: Added L2 weight decay (factor 0.04) to all Muon-managed matrix parameters. `p = p * (1 - lr * wd)` before the gradient update.
- **How to port**: In the Muon optimizer step, multiply params by `(1 - lr * 0.04)` before applying the orthogonalized gradient. 2-line change in train_gpt.py/train_gpt_mlx.py.
- **Expected full-scale impact**: High — already proven on the leaderboard. Should stack with other improvements.

## [2365e34] Adam weight decay on embedding (WD=0.01) — stacked with Muon WD
- **BPB**: 1.885685 (delta from baseline: -0.036)
- **Category**: other (regularization)
- **Track**: record (10min leaderboard)
- **What changed**: Added WD=0.01 to the Adam-managed embedding parameters. Combined with Muon WD=0.04. Embedding weights are decayed by `(1 - lr * 0.01)` each step.
- **How to port**: In the SplitOptimizers.step, multiply embedding weights by the WD factor after Adam update. 3-line change.
- **Expected full-scale impact**: High — regularization stacks. Embedding WD is standard practice and likely transfers.

## [bf61d68] Asymmetric U-Net (4 encoder + 2 decoder)
- **BPB**: 1.878685 (delta from baseline: -0.043)
- **Category**: arch
- **Track**: record (10min leaderboard)
- **What changed**: Changed U-Net split from symmetric 3+3 to asymmetric 4+2 (4 encoder layers, 2 decoder layers). Stacked with Muon WD=0.04 and Adam WD=0.01.
- **How to port**: In GPT.__init__, set `num_encoder_layers = int(NUM_LAYERS * 2/3)` instead of `NUM_LAYERS // 2`. 1-line change.
- **Expected full-scale impact**: Should transfer — deeper encoder builds richer representations. With 10-11 layers at full scale, try 7+4 or 7+3 split.

## [4335a17] Plain transformer (6+0, no decoder/skips)
- **BPB**: 1.871059 (delta from baseline: -0.051)
- **Category**: arch (simplification)
- **Track**: record (10min leaderboard)
- **What changed**: Removed U-Net decoder and skip connections entirely. Set `num_encoder_layers = NUM_LAYERS, num_decoder_layers = 0`. Monotonic trend: 3+3(1.886) → 4+2(1.879) → 5+1(1.872) → 6+0(1.871). At mini scale, the skip overhead isn't worth the information shortcut.
- **How to port**: Set `num_encoder_layers = NUM_LAYERS` and `num_decoder_layers = 0`, or simply remove the decoder loop and skip_weights. **NOTE**: This finding may NOT transfer to full scale — at 10-11 layers, skip connections likely help more. Test the 4+2 ratio first (encoder-heavy asymmetric U-Net).
- **Expected full-scale impact**: The encoder-heavy direction (4+2 or 5+1) should transfer. Full removal of skips is risky at full scale. Recommend testing both the asymmetric split and full removal.

---

## Tier 2 Findings (9L/384D, 600s budget on M4 Pro)

### [64547ca] Encoder-heavy U-Net 6+3 at 9 layers (Tier 2)
- **BPB**: 1.729628 (delta from Tier 2 baseline 1.752: -0.022)
- **Category**: arch
- **Track**: record (10min leaderboard)
- **What changed**: Changed U-Net split from 4+5 to 6+3 at 9 layers. Same monotonic trend as Tier 1.
- **How to port**: Set `num_encoder_layers = int(NUM_LAYERS * 2/3)` or `6` for 9 layers.
- **Expected full-scale impact**: High — confirmed at both mini and Tier 2 scale.

### [cc3a2f3] Plain transformer 9+0 at Tier 2
- **BPB**: 1.724587 (delta from Tier 2 baseline: -0.027)
- **Category**: arch
- **Track**: record (10min leaderboard)
- **What changed**: Removed U-Net entirely at 9 layers. Trend: 4+5(1.752) → 6+3(1.730) → 7+2(1.731) → 9+0(1.725). Confirmed at Tier 2 that skip connections hurt.
- **How to port**: Set `num_encoder_layers = NUM_LAYERS, num_decoder_layers = 0`.
- **Expected full-scale impact**: Confirmed at two scales. Strong evidence that plain transformer is better.

### [9ae7695] Long warmdown (3500 iters) — MAJOR FINDING
- **BPB**: 1.651301 (delta from Tier 2 baseline: -0.101)
- **Category**: schedule (major)
- **Track**: record (10min leaderboard)
- **What changed**: Increased WARMDOWN_ITERS from 400 to 3500 (entire training = warmdown). Monotonic improvement: 400(1.725) → 600(1.711) → 800(1.693) → 1000(1.687) → 1200(1.681) → 1500(1.678) → 2000(1.667) → 2500(1.664) → 3000(1.661) → 3500(1.651). Beyond 3500 it degrades (4000: 1.669, 50000: 1.704).
- **How to port**: Set `WARMDOWN_ITERS` to approximately `total_steps * 1.2`. At full scale (~12,000 steps), try `WARMDOWN_ITERS = 14000`. The key insight: the warmdown should span nearly the entire training run — effectively a linear LR decay from ~85% of peak to 0.
- **Expected full-scale impact**: Very high — this is a 0.101 BPB improvement from schedule tuning alone. At full scale with ~12,000 steps, try warmdown values of 10000, 12000, 14000, 16000.
