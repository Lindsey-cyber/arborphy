# Trial Process

Each experiment should be treated as a small trial with its own setup, output
artifact, notes, and commit.

## Required Flow

1. `git checkout X`
   - Start from the parent branch for the trial setup.

2. Work on experiment setup
   - Make only the code/config changes needed for the trial.
   - Keep data expansion and unrelated refactors out of the trial.

3. `git checkout X.x`
   - Move to the specific trial branch or checkpoint before running.

4. Run experiment
   - Use the trial wrapper when possible: `uv run python scripts/run_stepwise_trial.py`.
   - Let the runner auto-generate a distinct output filename, or set `EXPERIMENT_OUT_FILE` when a fixed name is needed.
   - Record the exact command in the trial log.

5. Gather artifacts
   - Record output CSV path.
   - Record image URL(s).
   - Record exact prompt(s) sent to the model.
   - Record raw model answer(s).
   - Record parsed answer(s).
   - Record the parse rule used.
   - Record expected answer and correctness fields.

6. Log results
   - Update the trial log in `trials/`.
   - Summarize what worked, what failed, and what changed.

7. Commit
   - Commit only the trial setup, trial log, and intended artifacts.
   - Do not include unrelated local/user changes.

## Trial Log Fields

- Trial ID
- Parent branch / setup branch
- Run branch / checkpoint
- Goal
- Exact command
- Output artifact path
- Image URL
- Prompt parts JSON
- Input species / observation
- Expected answer
- Raw model answer
- Parsed answer
- Parse rule
- Correctness
- Blockers
- Next step
