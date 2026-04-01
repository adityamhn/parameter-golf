# mini-golf

Autonomous research for the OpenAI Parameter Golf challenge on a single RTX 3060.

## The Challenge

**Parameter Golf**: train the best language model that fits in a **16 MB artifact** (code + compressed weights) within **10 minutes on 8×H100s**. Metric is **val_bpb** (bits per byte on FineWeb validation set) — lower is better.

Hard constraints:
- **16,000,000 byte** artifact limit: code + int6-quantized lzma-compressed model weights
- Any architectural change that helps BPB must keep artifact ≤ 16 MB
- Eval uses a 10-minute TTT pass on 8×H100s (not run here — proxy metric is raw val_bpb)

Current leaderboard top: **1.1194 BPB** (LeakyReLU² + Legal Score-First TTT + Parallel Muon).
Our proxy baseline after Tier 1 research: **1.3354 BPB** (12L, KV2, ROPE23, LEAKY0.72).

## Setup

You are already on the research branch. Do the following once at the start:

1. **Read in-scope files** (context only — do not modify `prepare` or data):
   - `README.md` — challenge context, leaderboard, wishlist
   - `train_gpt_single_gpu.py` — **the only file you edit**
2. **Verify data**: check that `./data/datasets/fineweb10B_sp1024/` exists and contains `.bin` shards. If missing, tell the human.
3. **Check results.tsv**: it already has prior results. The current best is **1.3354 BPB**.
4. **Confirm and start the loop immediately.**

## Experimentation

Each experiment runs on a single RTX 3060 (12 GB VRAM). Fixed budget per run:

```bash
GRAD_ACCUM_STEPS=8 VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=100 \
MAX_WALLCLOCK_SECONDS=0 TTT_ENABLED=0 EVAL_STRIDE=0 SEED=1337 \
PYTHONUNBUFFERED=1 .venv/bin/python3 train_gpt_single_gpu.py > run.log 2>&1
```

(All other hyperparameters — 2000 iters, 98304 tokens/step, 12L, KV2, ROPE23, LEAKY0.72 — are now the defaults in the file. Override via env vars only when testing changes.)

**What you CAN do:**
- Modify `train_gpt_single_gpu.py` freely — architecture, optimizer, hyperparameters, training loop.
- Change any hyperparameter default in the `Hyperparameters` class.
- Add new architectural components inside the file (new classes, functions).

**What you CANNOT do:**
- Modify data loading or the evaluation harness (`val_bpb` computation).
- Install new packages not already in `.venv/`.
- Exceed ~10 GB VRAM (hard 12 GB total; keep headroom).
- Push artifact above 16 MB — check the printed `Total submission size int6+lzma:` line.

**The goal: get the lowest val_bpb while keeping artifact ≤ 16,000,000 bytes.**

**Simplicity criterion**: A tiny gain from ugly complexity is not worth it. Deleting code and matching or improving BPB is a great win. Weigh complexity cost against improvement magnitude.

## Output format

After each run, grep the key metrics:

```bash
grep -E "val_bpb|peak memory|Total submission size" run.log | tail -10
```

The run prints lines like:
```
step  500 | val_bpb 1.3701
step 1000 | val_bpb 1.3520
step 1500 | val_bpb 1.3420
step 2000 | val_bpb 1.3354
peak memory allocated: 3431 MiB
Total submission size int6+lzma: 16132749 bytes
```

A crash (OOM, Python exception) will produce no `val_bpb` line. Run `tail -n 50 run.log` to see the traceback.

## Logging results

Record every run in `results.tsv` (tab-separated, NOT comma-separated).

Columns:
```
commit	val_bpb	memory_gb	artifact_mb	status	description
```

1. short git commit hash (7 chars)
2. final val_bpb (use 0.000000 for crashes)
3. peak memory in GB (divide MiB by 1024, round to .1f)
4. artifact size in MB (divide bytes by 1,000,000, round to .2f — use 0.00 for crashes)
5. status: `keep`, `discard`, or `crash`
6. short description of the change

Example:
```
commit	val_bpb	memory_gb	artifact_mb	status	description
40a50ed	1.3354	3.4	16.13	keep	best stack — 12L KV2 ROPE23 LEAKY0.72 — current best
```

**Do NOT git-commit results.tsv.** Leave it untracked.

## Ideas to explore

Use these as a starting menu. Exhaust obvious wins first, then go weirder.

### Optimizer & training dynamics
- Muon momentum schedule tuning (start 0.85→0.99 vs current 0.92→0.99)
- Lower scalar / tied-embed LR to reduce overfitting at end of warmdown
- Gradient clipping threshold (try 0.2, 0.4)
- Weight decay sweep (MUON_WD, ADAM_WD: 0.01, 0.02, 0.06)
- Cosine vs trapezoidal warmdown shape

### Architecture: attention
- Reduce to 1 KV head (MQA) — saves params → smaller artifact
- Sliding window attention on early layers, full on late layers
- RoPE dims sweep (20, 24, 26, 32)
- Increase rope_base (50000, 100000 — better long-range)
- ALiBi or no positional encoding on some layers

### Architecture: MLP / activations
- LeakyReLU² slope sweep (0.5, 0.6, 0.8, 1.0)
- SwiGLU (gated activation, standard in LLaMA) — replaces LeakyReLU²
- GeGLU
- MLP multiplier sweep (2.5, 3.5, 4.0)

### Architecture: macro
- Depth vs width: try 13L at narrower dim, or 10L at wider
- Parallel attention + MLP (PaLM-style: compute attn and MLP on same residual, sum)
- Cross-layer KV sharing (share KV between adjacent layers)
- Parameter sharing / weight tying across alternating layers
- Mixture of Depths: skip MLP on some tokens (top-k routing on residual norm)
- Universal transformer: cycle through a single shared layer N times

### Regularization & normalization
- RMSNorm scale init sweep (LN_SCALE: 0.8, 1.2)
- QK normalization (normalize Q and K before dot product — stabilizes training)
- Logit soft-cap sweep (20.0, 40.0, off)
- Dropout on attention weights (tiny: 0.05)

### Embedding & vocab
- Bigram vocab size sweep (1024, 2048) — trades artifact bytes vs expressivity
- Bigram dim sweep (64, 256)
- Untie embeddings if it helps (adds params → watch artifact size)

### OpenAI wishlist (from README)
- **JEPA**: predict latent representations of future tokens instead of raw tokens
- **Text diffusion**: replace autoregressive head with diffusion objective on the last N tokens
- **H-net tokenization**: byte-level model with hierarchical token merging
- **Universal transformer / depth recurrence**: single shared layer looped N times
- **State-space model layers**: replace some attention heads with Mamba/S4/RWKV-style recurrence
- **Megakernels**: fused CUDA kernel for attn+MLP in one pass (if nvcc available)
- **Learning adapters on random linear maps**: random projection adapters in each layer

### Quantization-aware
- INT4 weights with higher capacity (more layers/width, same artifact)
- Mixed precision: INT4 for MLP, INT8 for attention

## The experiment loop

LOOP FOREVER:

1. Review `results.tsv` and the current branch HEAD — know what the current best BPB is.
2. Pick the highest-potential untried idea (or a follow-up to a near-miss).
3. Modify `train_gpt_single_gpu.py` directly.
4. `git add train_gpt_single_gpu.py && git commit -m "experiment: <short description>"`
5. Run: `GRAD_ACCUM_STEPS=8 VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=100 MAX_WALLCLOCK_SECONDS=0 TTT_ENABLED=0 EVAL_STRIDE=0 SEED=1337 PYTHONUNBUFFERED=1 .venv/bin/python3 train_gpt_single_gpu.py > run.log 2>&1`
6. Check results: `grep -E "val_bpb|peak memory|Total submission size" run.log | tail -10`
7. If empty → crash. Run `tail -n 50 run.log`, fix if trivial, otherwise discard and `git reset --hard HEAD~1`.
8. Log the result to `results.tsv`.
9. If val_bpb improved **and** artifact ≤ 16,000,000 bytes → keep the commit, advance the branch.
10. If val_bpb is worse, equal, or artifact > 16 MB → `git reset --hard HEAD~1` (discard commit).
11. GOTO 1.

**Timeout**: if a run exceeds 80 minutes wall clock, kill it (`kill <pid>`) and treat as crash.

**Crashes**: Fix trivial bugs and re-run. If the idea is fundamentally broken, log `crash` and move on.

**NEVER STOP**: Do NOT pause to ask the human whether to continue. Do NOT say "should I keep going?". You are autonomous. Keep running experiments until the human interrupts you. If you run out of ideas, re-read the ideas list above, combine near-misses, or try more radical changes. The loop runs until manually stopped, period.
