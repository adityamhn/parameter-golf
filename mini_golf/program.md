# mini-golf

This is an experiment to have the LLM do its own research on the Parameter Golf challenge.

## The Challenge

**Parameter Golf** is an OpenAI challenge to train the best language model that fits in a **16MB artifact** (code + compressed model) and trains in **under 10 minutes on 8×H100s**. Evaluation metric is **val_bpb** (bits per byte on FineWeb validation set) — lower is better, tokenizer-agnostic.

Key constraints that affect architecture decisions:
- **16,000,000 byte artifact limit**: code bytes + int6-quantized lzma-compressed model weights. More parameters = harder to fit. The current SOTA uses 26.9M params → ~4.6MB compressed. There's headroom, but not unlimited.
- **10-minute training budget** on 8×H100: ~7,185 steps at 83ms/step. Our proxy runs 700 steps at ~1.3s/step on a single RTX 3060.
- **10-minute eval budget**: sliding window eval (stride 64) + legal test-time training (TTT). Architecture choices affect eval-time perf too.
- **Evaluation at any sequence length** is allowed. Longer context during eval can improve BPB.

## Leaderboard (as of March 30, 2026)

| Score | Summary |
|------:|---------|
| **1.1194** | **LeakyReLU² + Legal TTT + Parallel Muon** (current SOTA — our baseline) |
| 1.1228 | 11L EMA + GPTQ-lite + warmdown3500 |
| 1.1248 | 11L Partial RoPE + LN Scale + EMA + XSA4 |
| 1.1271 | 11L XSA4 + EMA + Int6 MLP3x |
| 1.1307 | 11L Efficient Partial XSA |
| 1.1428 | 10L Int5-MLP + BigramHash(10240) |
| 1.1458 | Int6 MLP3x + SmearGate + BigramHash |
| 1.1502 | 11L MLP3x + Int6 QAT |
| 1.1556 | SmearGate + OrthoInit + Muon WD |
| 1.1570 | Ternary Quantization (73.7M params, 1/0/-1) |
| 1.1928 | LoRA TTT |
| 1.2244 | Naive Baseline (9L 512D 1024vocab) |

The jump from 1.2244 → 1.1194 came from stacking ~10 independent techniques. Each individual technique contributed 0.001–0.01 BPB. The biggest single wins were: sliding window eval (~0.03), 3× MLP (~0.02), deeper model (~0.01), EMA (~0.01).

### OpenAI's wish list (frontier ideas they want explored)

- [ ] JEPA (Joint Embedding Predictive Architecture)
- [ ] Text diffusion
- [ ] Universal transformer (depth recurrence with single block)
- [ ] Megakernels
- [ ] State-space models, E2E TTT, super long context
- [ ] Learning adapters on random linear maps
- [x] 1-bit quantization (done, 1.1239 BPB non-record)
- [x] Ternary quantization (done, 1.1570 BPB)

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar30`). The branch `mini-golf/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b mini-golf/<tag>` from current main.
3. **Read the in-scope files**: The project is small. Read these files for full context:
   - `mini_golf/program.md` — this file, the research constitution.
   - `train_gpt_single_gpu.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `data/datasets/fineweb10B_sp1024/` contains `fineweb_train_*.bin` and `fineweb_val_*.bin`, and `data/tokenizers/fineweb_1024_bpe.model` exists.
5. **Initialize results.tsv**: Create `mini_golf/results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single RTX 3060 12GB GPU. The training script runs for a **fixed number of iterations (700)** with a reduced batch size (98,304 tokens/step instead of the full 786,432). This gives ~15 minutes per experiment. You launch it like this:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python3 train_gpt_single_gpu.py > run.log 2>&1
```

All hyperparameters are controlled via environment variables. The **SOTA baseline** (1.1194 BPB on 8×H100) uses this exact run command:

```bash
ITERATIONS=700 WARMUP_STEPS=10 WARMDOWN_ITERS=350 \
TRAIN_BATCH_TOKENS=98304 GRAD_ACCUM_STEPS=8 \
VAL_LOSS_EVERY=350 TRAIN_LOG_EVERY=50 \
MAX_WALLCLOCK_SECONDS=0 TTT_ENABLED=0 EVAL_STRIDE=0 \
NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3.0 \
BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
LOGIT_SOFTCAP=30.0 TIE_EMBEDDINGS=1 TIED_EMBED_INIT_STD=0.005 \
QK_GAIN_INIT=1.5 GRAD_CLIP_NORM=0.3 \
SEED=1337 PYTHONUNBUFFERED=1 \
.venv/bin/python3 train_gpt_single_gpu.py > run.log 2>&1
```

The env vars that control the proxy budget (ITERATIONS, TRAIN_BATCH_TOKENS, GRAD_ACCUM_STEPS, WARMDOWN_ITERS, WARMUP_STEPS, VAL_LOSS_EVERY, TRAIN_LOG_EVERY, MAX_WALLCLOCK_SECONDS, TTT_ENABLED, EVAL_STRIDE) are FIXED — do not change them between experiments. Everything else is fair game.

**What you CAN do:**
- Modify `train_gpt_single_gpu.py` — this is the only file you edit. Architecture, optimizer, activations, attention, embeddings, everything is fair game.
- Add or change env vars for architectural hyperparameters (NUM_LAYERS, MLP_MULT, etc.) in your run command.

**What you CANNOT do:**
- Change the proxy budget env vars (ITERATIONS, TRAIN_BATCH_TOKENS, GRAD_ACCUM_STEPS, etc.)
- Install new packages or add dependencies.
- Modify any other file (data loading, evaluation are baked into train_gpt_single_gpu.py).

**The goal is simple: get the lowest val_bpb.** Since the iteration count and batch size are fixed, you're comparing architectures and techniques fairly. Everything is fair game: change the model architecture, activations, attention mechanism, embeddings, optimizer settings, etc.

**The first run**: Your very first run should always be the SOTA baseline command above, unmodified. This establishes the comparison point.

## SOTA techniques (what the baseline already includes)

The baseline `train_gpt_single_gpu.py` already implements all current leaderboard techniques:

- **LeakyReLU(0.5)²** activation: `F.leaky_relu(x, 0.5).square()` instead of `relu(x).square()`. Preserves negative gradient flow, eliminates dead neurons.
- **Parallel Muon** optimizer: Contiguous 3D parameter banks + batched Newton-Schulz orthogonalization. Replaces per-layer DDP with reduce-scatter/all-gather.
- **11L/512D/8H/4KV** with 3× MLP width and GQA (grouped-query attention).
- **BigramHash** (1536 vocab): Hash-based bigram features as auxiliary input, gives the model character-level context cheaply.
- **XSA** (cross-sequence attention) in last 4 layers: Attention across sequence boundaries in deepest layers.
- **Partial RoPE** (16/64 dims): Only apply rotary position embeddings to 16 of 64 head dimensions.
- **LN Scale** (1/√(layer+1)): Layer-dependent normalization scaling.
- **Value Embeddings** (VE128 on layers 9-10): Extra learned value vectors in the deepest attention layers.
- **EMA(0.997) + SWA(every 50)**: Weight averaging — exponential moving average plus stochastic weight averaging.
- **Late QAT**: Quantization-aware training with straight-through estimator, activated late in training.
- **Legal score-first TTT**: Test-time training that adapts on already-scored validation tokens (disabled in proxy for speed).

Your job is to find improvements **on top of** this stack.

## Output format

The script prints step-by-step logs. The key metric line looks like:

```
step:700/700 val_loss:X.XXXX val_bpb:X.XXXX train_time:XXXXXms step_avg:XXXX.XXms
peak memory allocated: XXXX MiB reserved: XXXX MiB
```

Extract the results:

```bash
grep "val_bpb\|peak memory" run.log | tail -5
```

## Logging results

When an experiment is done, log it to `mini_golf/results.tsv` (tab-separated).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 2.345678) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (divide MiB by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	2.345678	3.3	keep	SOTA baseline
b2c3d4e	2.312345	3.4	keep	SwiGLU activation instead of LeakyReLU²
c3d4e5f	2.400000	3.3	discard	13 layers (slower convergence at 700 steps)
d4e5f6g	0.000000	0.0	crash	double MLP width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `mini-golf/mar30`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Review `mini_golf/results.tsv` to understand what's been tried.
3. Hack `train_gpt_single_gpu.py` with an experimental idea.
4. `git commit -am "category: description of the change"`
5. Run the experiment: `ITERATIONS=700 WARMUP_STEPS=10 WARMDOWN_ITERS=350 TRAIN_BATCH_TOKENS=98304 GRAD_ACCUM_STEPS=8 VAL_LOSS_EVERY=350 TRAIN_LOG_EVERY=50 MAX_WALLCLOCK_SECONDS=0 TTT_ENABLED=0 EVAL_STRIDE=0 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3.0 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 LOGIT_SOFTCAP=30.0 TIE_EMBEDDINGS=1 TIED_EMBED_INIT_STD=0.005 QK_GAIN_INIT=1.5 GRAD_CLIP_NORM=0.3 SEED=1337 PYTHONUNBUFFERED=1 .venv/bin/python3 train_gpt_single_gpu.py > run.log 2>&1` (you may override architectural env vars like NUM_LAYERS, MLP_MULT, etc. — but never the proxy budget vars)
6. Read out the results: `grep "val_bpb\|peak memory" run.log | tail -5`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
8. Record the results in `mini_golf/results.tsv` (NOTE: do not commit results.tsv, leave it untracked by git)
9. If val_bpb improved (lower), you "advance" the branch, keeping the git commit.
10. If val_bpb is equal or worse, you `git reset --hard HEAD~1` to revert.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take ~15 minutes total (700 steps × ~1.3s + warmup/eval overhead). If a run exceeds 25 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, etc.), use your judgment: if it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes ~15 minutes then you can run approx 4/hour, for a total of about 30 over an 8-hour sleep. The user then wakes up to `mini_golf/results.tsv` — the complete research output.

## Queued experiment families (after ROPE/LEAKY ridge stalls)

Prioritize cheap local **ROPE_DIMS** / **LEAKY_SLOPE** sweeps around the current best until improvements stop. **Odd** partial `ROPE_DIMS` are allowed: RoPE applies to the leading `ROPE_DIMS-1` dimensions and one dimension is left unrotated within the rope prefix. Then draw from this queue (small, parameter-efficient changes only; edit `train_gpt_single_gpu.py`):

1. **Lightweight recurrence / stateful mixing (xLSTM-style, not a full swap):** shallow recurrent mixing in the last few layers; a small recurrent token mixer or a shared recurrent block applied with minimal extra params; avoid blowing the 16MB artifact budget.

2. **Parameter-efficient mixtures / adapters (Phi-4-Mini / MoLoRA-style):** tiny gated low-rank adapters or expert-style modulation inside MLP and/or attention; keep rank and expert count tiny so total params stay flat-ish.

3. **Activation-triggered / token-conditional adapters (Activated LoRA-style):** small residual adapters or scales gated by activations or simple token-conditional paths; no reliance on KV-cache semantics—must train and eval in this script as-is.

4. **Structured feature routing (interpretability-inspired):** per-layer selective residual routing, light sparse feature gates, or clearer early (lexical) vs late (value/reasoning) pathway separation without generic width increases.

## Research strategy

Since the baseline already stacks all known leaderboard techniques, you need to find **novel improvements**. Prioritize by expected impact:

**High priority (proven ingredients at scale, but may have room to tune or combine differently):**
- Activation variants: different LeakyReLU slopes (0.3, 0.7), SwiGLU, GELU×tanh, gated activations
- Depth: 12-13 layers (may need to check the 16MB compressed size — `grep "int6+lzma" run.log`)
- MLP: 4× width, gated MLP (SwiGLU-style), shared MLP across layers
- Attention: more XSA layers, different RoPE dim splits, different head/KV-head ratios

**Medium priority (less explored but plausible):**
- Different norm strategies (post-norm, sandwich norm)
- Different weight averaging (LAWA, different EMA decays, SWA frequency)
- Embedding tricks: larger BigramHash vocab, VE on more layers, different VE dims
- Better initialization schemes

**Frontier (from OpenAI's wish list — even a sign-of-life is valuable):**
- Depth recurrence / universal transformer (run same block N times)
- State-space layers (replace some attention layers with SSM)
- Mixture-of-experts (replace some MLPs with sparse experts)
- Any creative architectural idea you can think of

**Watch out for:**
- OOM on 12GB VRAM — the baseline uses ~3.3 GB, you have headroom but not infinite
- Compressed model size — check `Serialized model int6+lzma` in the log stays under 16MB
- Changes that help early in training but hurt at convergence (common trap at 700 steps)
