# mini-golf

This is an experiment to have the LLM do its own research for parameter golf on a single RTX 3060.

## Setup

To set up a new run, do this once:

1. Create or switch to a dedicated branch such as `mini-golf/<tag>`.
2. Read these files for full context:
   - `README.md` — challenge context, leaderboard, and wishlist
   - `mini_golf/results.tsv` — memory of all prior experiments
   - `train_gpt_single_gpu.py` — the only file you edit
3. Verify data exists in `data/datasets/fineweb10B_sp1024/`. If it is missing, tell the human.
4. Confirm setup looks good, then start the loop immediately.

## Experimentation

Each experiment runs on a single GPU with a fixed proxy budget. Launch it exactly like this:

```bash
GRAD_ACCUM_STEPS=8 VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=100 \
MAX_WALLCLOCK_SECONDS=0 TTT_ENABLED=0 EVAL_STRIDE=0 SEED=1337 \
PYTHONUNBUFFERED=1 uv run python train_gpt_single_gpu.py > run.log 2>&1
```

The training file already contains the current working defaults. Change them directly in `train_gpt_single_gpu.py` when you test a new idea.

What you CAN do:
- Modify `train_gpt_single_gpu.py` freely.
- Change architecture, optimizer, hyperparameters, schedules, and defaults.
- Add new helper functions or modules inside that file if needed.

What you CANNOT do:
- Modify the evaluation harness or data loading logic.
- Install new packages or dependencies.
- Edit files other than `train_gpt_single_gpu.py` during the research loop.

The goal is simple: get the lowest `val_bpb`.

Artifact size and VRAM still matter because this is parameter golf, but for this local research loop they are secondary diagnostics. Log them every time. If a run meaningfully improves `val_bpb` but is slightly over 16 MB, you may still keep it as the current research champion and then try to recover bytes in follow-up experiments.

Simplicity criterion: all else being equal, simpler is better. A tiny improvement from a messy change is not worth much. A tiny improvement from a clean change, or the same score with less complexity, is a good outcome.

## Reading Results

After every run, inspect `run.log` and `mini_golf/results.tsv`.

Extract the key metrics from `run.log`:

```bash
rg "val_bpb|peak memory|Total submission size" run.log
```

You should always know:
- the final `val_bpb`
- the peak memory in MiB
- the artifact size in bytes

If the run crashes, inspect the end of `run.log`, fix trivial bugs, and try again. If the idea itself is bad or unstable, log it as a crash or discard and move on.

Most importantly: after each experiment, think before acting again.

Use `results.tsv` as your memory:
- Read the recent rows before choosing the next idea.
- Notice which directions helped, which failed, and which combinations are worth following up.
- Prefer one change at a time until you see real signal.
- When something improves, do 1-3 nearby follow-ups before jumping to a totally different area.
- If an area keeps failing, pivot.

Do not run blind sweeps forever. Use the results to guide the next move.

## Logging Results

When an experiment is done, log it to `mini_golf/results.tsv` as tab-separated values.

The TSV columns are:

```text
commit	val_bpb	memory_gb	artifact_mb	status	description
```

1. short git commit hash
2. final `val_bpb` achieved, or `0.000000` for crashes
3. peak memory in GB, rounded to `.1f`
4. artifact size in MB, rounded to `.2f`
5. status: `keep`, `discard`, or `crash`
6. short description of what the experiment tried

Example:

```text
40a50ed	1.3354	3.4	16.13	keep	best stack — 12L KV2 ROPE23 LEAKY0.72
```

Do not git-commit `mini_golf/results.tsv`.

## Research Priorities

Use the evidence in `results.tsv` first. Then use the leaderboard and wishlist for new directions.

Priority 1: follow up on ideas that already showed local signal
- `MUON_MOMENTUM_WARMUP_START=0.88`
- nearby optimizer schedule changes
- combinations built around the strongest recent local winner

Priority 2: test the training-side parts of the current leaderboard top that are measurable locally
- XSA on all layers
- larger BigramHash settings around `3072 x 112`
- dropping TTT if it is still active anywhere
- combinations of the above with your best local stack

Priority 3: broader model ideas from `README.md`
- alternative activations or gated MLPs
- RoPE or positional changes
- depth/width tradeoffs
- parameter sharing or recurrence
- state-space or universal-transformer style changes
- any other promising idea from the challenge wishlist

Low priority for local proxy runs:
- AR self-generated GPTQ calibration
- Full Hessian GPTQ
- final compression tuning
- other post-training quantization-only improvements

These may matter for the real leaderboard, but they are not the best first target for this local `val_bpb` loop.

## The Experiment Loop

The run happens on a dedicated branch and continues indefinitely.

LOOP FOREVER:

1. Read the current git state and review `mini_golf/results.tsv`.
2. Identify the current research champion and the most promising next direction.
3. Edit `train_gpt_single_gpu.py` with one new idea or a tight follow-up to a recent success.
4. Commit the change.
5. Run the experiment with the fixed command above, redirecting output to `run.log`.
6. Read `run.log`, extract `val_bpb`, memory, and artifact size.
7. Record the result in `mini_golf/results.tsv`.
8. Think about what happened.
9. If `val_bpb` improved, keep the commit and continue building from it.
10. If `val_bpb` is equal or worse, revert to the previous champion commit.
11. Repeat.

If the run crashes:
- fix trivial issues and retry
- if the idea is fundamentally bad, record it and revert

Timeout:
- if a run takes far longer than expected, kill it and treat it as a crash

The first run:
- if `results.tsv` is empty, run the current file once to establish a baseline
- if `results.tsv` already has results, do not waste a run repeating the baseline unless you suspect the code has changed and the baseline is no longer trustworthy

NEVER STOP:
- once the loop starts, do not pause to ask the human whether to continue
- do not ask for permission after each run
- keep reading the results, thinking, and improving until the human interrupts you
