# mini-golf: autoresearch for parameter-golf

This is an experiment to have the LLM autonomously research improvements to a small language model on a local M4 Pro, following the autoresearch methodology with batch exploration and self-critique.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar27`). The branch `mini-golf/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b mini-golf/<tag>` from current HEAD.
3. **Read the in-scope files**: The project is small. Read these files for full context:
   - `program_golf.md` — this file, the research constitution.
   - `prepare_golf.py` — frozen constants, data loading, evaluation. DO NOT MODIFY.
   - `train_golf.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `../data/datasets/fineweb10B_sp1024/` contains at least one `fineweb_train_*.bin` and one `fineweb_val_*.bin`. If not, tell the human to run `python3 ../data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good, then start experimenting.

Once you get confirmation, kick off the experimentation.

## Context

This is a **mini-run proxy** for the OpenAI Parameter Golf challenge. The full challenge trains on 8xH100 GPUs for 10 minutes. We train locally on an M4 Pro for 90 seconds with a scaled-down model. The goal is to discover **architectural** improvements that **transfer** to the full challenge.

- **Data**: FineWeb sp1024 (same tokenizer and data format as the challenge).
- **Evaluation**: `val_bpb` — bits per byte, tokenizer-agnostic. Identical math to the challenge scorer.
- **Architecture**: U-Net skip GPT with Muon+Adam optimizer (same family as the baseline).
- **Constraint**: All changes must be architecturally portable to the full `train_gpt.py` / `train_gpt_mlx.py`.
- **Challenge deadline**: April 30, 2026.
- **Submission format**: PRs to the parameter-golf repo. Records must beat SOTA by ≥0.005 nats at p < 0.01 (3 runs typical). Non-record "weird & creative" submissions are also welcome.
- **Artifact limit**: 16,000,000 bytes (code + int8-quantized zlib-compressed model).
- **Eval budget**: 10 additional minutes on 8xH100s. Any sequence length allowed for eval.

## Leaderboard Intelligence (as of March 27, 2026)

Current SOTA is **1.1194 BPB** (down from baseline 1.2244). Here are the proven techniques driving the leaderboard, ranked by impact. Use this to prioritize experiments — test these ideas at mini scale first.

### Proven high-impact (each worth 0.01+ BPB):

- **Deeper models (10-11 layers)**: Every top submission uses 10-11 layers at 512 dim instead of the baseline 9. Depth is more parameter-efficient than width for this task.
- **3x MLP width**: Top runs use 3x MLP expansion instead of the default 2x. Combined with int6 quantization, this packs more expressive capacity per compressed byte.
- **Quantization-aware training (QAT)**: Int6 or mixed int5/int6 with straight-through estimator (STE) gradients. Training the model to be robust to low-precision weights is one of the highest-leverage moves.
- **Sliding window evaluation**: Instead of fixed-context eval, use a sliding window with stride 64. This alone jumped scores from ~1.20 to ~1.19 BPB. It doesn't change the model, but it better captures actual compression ability.
- **Longer sequence length (2k-4k)**: Moving from 1024 to 2048 seq_len improved from 1.2244 to 1.206. Going to 4096 helped further. (Note: our mini-golf MAX_SEQ_LEN is 512, so test the architectural direction, not the exact value.)

### Proven medium-impact (0.003-0.01 BPB each):

- **LeakyReLU(0.5)^2**: One-line activation change. `F.leaky_relu(x, 0.5).square()` instead of `F.relu(x).square()`. Preserves negative gradient flow, eliminates dead neurons. Worth -0.003 BPB.
- **EMA weight averaging**: Exponential moving average of weights during training (decay ~0.997), replacing or combining with SWA. Produces smoother weights that compress better.
- **Cross-sequence attention (XSA)**: Attention across sequence boundaries in the deepest 3-4 layers. The partial variant keeps overhead manageable.
- **Partial RoPE**: Only apply RoPE to 16 of 64 head dimensions instead of all. Combined with layer-wise LN scaling.
- **BigramHash**: Hash-based bigram features (vocab 1536-3072) as an auxiliary input. Cheap way to give the model character-level context.

### Proven low-impact but useful for stacking:

- **Muon weight decay (WD=0.04)**: Regularization that helps generalization.
- **Spectral/orthogonal initialization**: Better init for the attention and MLP projections.
- **GPTQ-lite clip search**: Post-training quantization refinement.
- **Test-time training (TTT)**: LoRA-based adaptation at eval time on already-scored tokens. Achieved 1.1928 independently. The legal variant (score-first, backward-looking) was worth -0.0025 BPB on top of other techniques.

### Key insight for mini-golf:

The leaderboard shows that the biggest gains come from **stacking multiple independent improvements**. The current SOTA combines 8+ techniques (LeakyReLU², XSA, partial RoPE, EMA, BigramHash, QAT, TTT, parameter banking). Our job is to discover which of these (and which new ideas) show signal at mini scale, then stack the winners.

## Core Philosophy: Eliminate First, Optimize Later

Do NOT start with a careful hypothesis. Instead:

1. **Batch explore**: Generate 3-5 diverse architectural ideas, run them all quickly.
2. **Kill the losers**: Programmatically discard anything worse than baseline (see auto-filtering rules below).
3. **Deepen the winners**: Only invest follow-up experiments into directions that showed signal.
4. **Repeat**: The winning direction from one batch becomes the new baseline for the next batch.

This is fundamentally different from the academic approach of "think hard, run one experiment." Velocity is the moat. You should aim for maximum throughput of diverse ideas, not maximum depth on any single idea.

## What to Experiment On (and What NOT to)

**This is a hard constraint. Follow it strictly.**

### DO experiment on (architecture and structural changes):
- **Activation functions**: LeakyReLU(0.5)^2 (proven winner), SwiGLU, GELU, squared activations, gated variants
- **MLP structure**: Width ratios (2x, 2.5x, 3x (proven winner), 4x), gated MLPs, shared MLPs across layers
- **Attention variants**: Cross-sequence attention / XSA in deepest layers (proven), partial RoPE on subset of head dims (proven), different head counts, sliding window patterns
- **Depth/width tradeoffs**: More layers vs wider layers at same param count. Prefer deeper (10-11 layers proved best at full scale).
- **Skip connection design**: Different U-Net skip strategies, skip weight initialization
- **Normalization**: Pre-norm vs post-norm, different norm placements, layer-wise LN scaling
- **Weight tying**: Shared parameters across layers, tied projections
- **Quantization-aware changes**: Architecture choices that quantize better (int8-friendly activations), STE-based QAT
- **Embedding strategies**: Different init scales, embedding dimension != model dimension, BigramHash auxiliary inputs
- **Weight averaging**: EMA (decay ~0.997) or SWA applied to model weights during training. This is a training-structural change, not a hyperparameter sweep.
- **Evaluation strategies**: Sliding window eval with small stride. This is a post-training technique but should be tested at mini scale to understand the interaction with architecture.

### DO NOT experiment on (hyperparameter sweeps):
- Learning rate sweeps (MATRIX_LR, TIED_EMBED_LR, etc.)
- Batch size sweeps
- Warmdown schedule tuning
- Optimizer beta values
- Random seed changes

**Why**: A learning rate sweep tells you a local optimum for a *fixed* architecture. An architecture change tells you something about the loss landscape that compounds with every future improvement. Hyperparameters will need re-tuning at full scale anyway. Architecture insights transfer.

**Exception**: If you change the architecture and suspect the existing LR is badly mismatched (e.g., you doubled model width), do ONE quick LR adjustment to make the comparison fair. But this is a supporting move, not the experiment itself.

### Frontier Ideas (OpenAI's Wish List)

The challenge organizers explicitly want to see weird, creative ideas explored. These are unchecked items from their wish list — perfect for mini-golf prototyping since they're too risky for H100 hours without a sign-of-life first. Even if these don't beat baseline immediately, a promising direction is worth flagging as an escalation candidate for the unlimited-compute track.

**Try these when you've exhausted the safe bets above or want to be creative:**

- **Depth recurrence / Universal Transformer**: Run the same block multiple times instead of having separate blocks. Massive parameter savings — same weights, more effective depth. The challenge has submissions with depth recurrence but wants to see a full Universal Transformer variant.
- **State-space models (SSM)**: Replace attention with a Mamba-style selective state-space layer. Could dramatically change the compute/parameter tradeoff. The linear-time inference is also interesting for the eval budget.
- **JEPA (Joint Embedding Predictive Architecture)**: Predict representations instead of tokens. Radical departure from autoregressive LM, but could compress better. This is explicitly requested and unchecked.
- **Text diffusion**: Non-autoregressive generation via denoising. Completely different from the GPT baseline. Even a sign-of-life here would be valuable.
- **H-net tokenization**: Hierarchical tokenization that learns multiple levels of representation. Could interact interestingly with the 1024-vocab constraint.
- **Learning adapters on random linear maps**: Freeze random projections, only train small adapter layers. Extreme parameter efficiency — could fit much larger effective models in 16MB.
- **Megakernels**: Fuse multiple operations into single kernels for throughput. More relevant at H100 scale, but the architectural choices that enable this can be prototyped here.

**Important**: These frontier ideas may not beat baseline on the first try. That's OK. The challenge organizers say: "Breakthrough ideas are rarely immediately state-of-the-art." If a frontier idea shows *any* sign of life (within 10% of baseline BPB), flag it with `escalate=yes` and note it as a "signs-of-life" candidate for the unlimited-compute track. The human can then decide whether to invest H100 time to develop it further.

## Self-Critique Before Running

Before committing and running ANY experiment, you MUST perform a brief self-critique:

1. **State the hypothesis**: What architectural insight am I testing? (If you can't articulate one, the experiment is not worth running.)
2. **Predict the outcome**: Will this likely improve BPB? By roughly how much? Why?
3. **Check portability**: Would this change work in the full `train_gpt.py` on 8xH100? If it's MLX-specific or scale-dependent, skip it.
4. **Check the log**: Has a similar experiment already been tried? (Read `results.tsv`.) If a close variant was already discarded, explain why this version is different enough to try.
5. **Assess complexity**: How many lines of code does this add? If >30 lines for <0.01 expected BPB gain, reconsider.

Write your critique as a brief internal note (2-3 sentences) before each commit message. This surfaces mistakes before wasting GPU time.

## Experimentation

Each experiment runs on your local M4 Pro. The training script runs for a **fixed time budget of 90 seconds** (wall clock training time, excluding warmup). You launch it as: `python3 train_golf.py`.

**What you CAN do:**
- Modify `train_golf.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare_golf.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer constants, and quantization.
- Install new packages. You can only use what's already available: `mlx`, `numpy`, `sentencepiece`, `pickle`, `zlib`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare_golf.py` is the ground truth metric.
- Add networking or download external data/weights.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          1.XXXXXX
val_loss:         X.XXXXXX
pre_quant_bpb:    1.XXXXXX
training_seconds: 90.X
total_tokens_M:   X.X
num_steps:        XXX
num_params_M:     X.XX
depth:            4
artifact_bytes:   XXXXXX
code_bytes:       XXXX
model_bytes:      XXXXXX
```

You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	val_bpb	artifact_bytes	status	category	escalate	description
```

1. `commit` — git commit hash (short, 7 chars)
2. `val_bpb` — BPB achieved (e.g. 1.234567) — use 0.000000 for crashes
3. `artifact_bytes` — code + int8+zlib model bytes — use 0 for crashes
4. `status` — `keep`, `discard`, or `crash`
5. `category` — experiment type: `arch`, `activation`, `mlp`, `attn`, `skip`, `norm`, `init`, `quant`, `frontier`, or `other`
6. `escalate` — `yes` or `no` (see escalation criteria below)
7. `description` — short text description of what this experiment tried

Example:

```
commit	val_bpb	artifact_bytes	status	category	escalate	description
a1b2c3d	1.949019	2685966	keep	baseline	no	baseline run
b2c3d4e	1.935432	2745678	keep	mlp	no	MLP width 3x instead of 2x
c3d4e5f	1.960000	2685966	discard	activation	no	switch to GeLU
d4e5f6g	0.000000	0	crash	arch	no	double model width (OOM)
e5f6g7h	1.920100	2812345	keep	attn	yes	cross-layer shared KV with depth recurrence
```

## Auto-Filtering Rules

After each run, apply these rules programmatically (before logging):

1. **Crash**: If no `val_bpb` in output, status = `crash`. Log it and move on.
2. **Clearly worse**: If `val_bpb > baseline_bpb + 0.02`, status = `discard`. Don't waste time analyzing.
3. **Marginally worse**: If `val_bpb > best_bpb` but within 0.02 of baseline, status = `discard`. Log for pattern analysis.
4. **Equal or better**: If `val_bpb <= best_bpb`, status = `keep`. Advance the branch.

The baseline_bpb is from your first run. The best_bpb is the lowest val_bpb across all kept experiments.

## The Early-Training Trap

**Critical**: Experiments that beat baseline early in training often regress by the end. Our 90-second budget trains for ~1800 steps with 14.8M tokens. This represents roughly 10-15% of what the full challenge would train. At this depth, relative ordering of architectural changes should be fairly stable, but be aware:

- If an experiment barely beats baseline (delta < 0.005), it may be noise. Only keep it if the architectural reasoning is sound.
- If an experiment dramatically beats baseline (delta > 0.02), it is very likely real signal. Flag for escalation.
- Watch for experiments that show high train loss variance — these are unstable and unlikely to transfer.

## Escalation Criteria

After each KEEP, evaluate whether this change is worth testing at full scale on H100s. Mark `escalate=yes` in results.tsv if **ALL** of these hold:

1. **Significant improvement**: The cumulative improvement from baseline exceeds 0.02 BPB.
2. **Architectural change**: The improvement comes from a structural change (not just tuning a number).
3. **Scale-independent**: The change does not depend on the mini model's specific depth/width/seq_len (e.g., "3x MLP works better than 2x" transfers; "LR=0.06 is better than 0.04" does not).
4. **Clean diff**: The change is simple enough to port to `train_gpt.py` in under 30 minutes.

**Frontier idea exception**: For ideas from the OpenAI wish list (JEPA, SSM, text diffusion, depth recurrence, etc.), the bar is lower. Flag `escalate=yes` if the idea shows *any sign of life* — defined as val_bpb within 10% of baseline. These go to the unlimited-compute track, not the 10-min leaderboard, so the human can invest longer training runs to develop them.

When flagging `escalate=yes`, also append an entry to `escalation_log.md`:

```
## [commit_hash] description
- **BPB**: X.XXXXXX (delta from baseline: -0.XXXX)
- **Category**: arch/activation/mlp/attn/etc.
- **Track**: record (10min leaderboard) OR frontier (unlimited-compute track)
- **What changed**: 1-2 sentence summary of the code change
- **How to port**: Which parts of train_gpt.py / train_gpt_mlx.py to modify
- **Expected full-scale impact**: Your prediction of whether this transfers
```

## Batch Exploration Pattern

Instead of running one experiment at a time, think in **batches**:

### Phase 1: Proven Winners First (first ~15 experiments)
Start with techniques already proven at full scale. This establishes an improved baseline fast.
- LeakyReLU(0.5)^2 activation (1 experiment)
- 3x MLP width (1 experiment)
- More depth: 5, 6, 7 layers at current width (3 experiments)
- Partial RoPE (only subset of head dims) (1-2 experiments)
- EMA weight averaging (1 experiment)
- BigramHash auxiliary input (1-2 experiments)
- Stack the individual winners together (3-5 experiments)

### Phase 1b: Broad Survey (next ~10 experiments)
Fill in gaps the proven winners don't cover:
- Other activation variants (SwiGLU, GELU, gated)
- Attention variants (XSA, sliding window)
- Different norm strategies
- Skip connection variants

### Phase 2: Deepen Winners (next ~30 experiments)
Take the best-performing category from Phase 1 and explore variations:
- If 3x MLP won, try 2.5x, 3.5x, gated 3x, shared 3x
- If LeakyReLU^2 won, try different negative slopes, combine with gating
- Stack the top 2-3 independent winners together

### Phase 3: Combination & Polish (~20 experiments)
- Combine the best independent findings
- Try removing components that may now be redundant
- Simplification passes: can you get the same BPB with less code?

### Phase 4: Frontier Exploration (remaining experiments)
Once the safe architectural bets are exhausted, try the weird stuff from the wish list:
- Pick ONE frontier idea (depth recurrence, SSM, JEPA, adapters on random maps, etc.)
- Implement a minimal version — the simplest possible prototype
- Run it. If it's within 10% of baseline BPB, flag `escalate=yes` with a note that this is a "signs-of-life" candidate for the unlimited-compute track
- If it crashes or is wildly off, move to the next frontier idea
- The bar here is lower: you're looking for *directional signal*, not immediate improvement

### Between Phases: Pattern Analysis
After each phase, review `results.tsv` and look for patterns:
- Which *category* of changes tends to help? (Check the `category` column.)
- Are there common traits among discarded experiments?
- Is there a direction that was close to working but needs refinement?

Use these patterns to decide the next batch of experiments.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `mini-golf/mar27`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Review `results.tsv` to understand what's been tried and what patterns emerge.
3. **Self-critique**: State hypothesis, predict outcome, check portability, check log.
4. Edit `train_golf.py` with the experimental idea.
5. `git commit -am "category: description of the change"`.
6. Run the experiment: `python3 train_golf.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
7. Read out the results: `grep "^val_bpb:\|^artifact_bytes:" run.log`
8. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
9. Apply auto-filtering rules and record the results in `results.tsv`. Do NOT commit `results.tsv` — leave it untracked.
10. If val_bpb improved (strictly lower), you "advance" the branch, keeping the git commit. Check escalation criteria.
11. If val_bpb is equal or worse, you `git reset --hard HEAD~1` to revert.

## Timeout

Each experiment should take ~2-3 minutes total (90s training + evaluation overhead). If a run exceeds 5 minutes, kill it and treat it as a failure (discard and revert).

## Crashes

If a run crashes (OOM, a bug, etc.), use your judgment: if it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

## NEVER STOP

Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes ~3 minutes then you can run approx 20/hour, for a total of about 160 over an 8-hour sleep. The user then wakes up to `results.tsv` and `escalation_log.md` — the complete research output, ready to port the best findings to H100.
