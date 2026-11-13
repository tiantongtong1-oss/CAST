# Fixed-checkpoint candidate scan

Run scan_candidates.py on GPU 0, with the same data/backbone as training.
Required: --checkpoint --source_path --target_path --output_dir.
Output directory must be new. Default --weights auto selects ema_teacher when
present, otherwise model; metadata records the choice. Override with --weights model.
This loads trusted checkpoints only; no optimizer, backward, or training-state resume.
No target labels affect candidates. Target TRAIN labels are exported for offline
analysis; do not repeatedly tune against this diagnostic set and claim held-out gains.

Outputs: candidates.csv, summary.txt, metadata.json. Includes teacher seven-class
probabilities, seven medoid distances, global source kNN votes, class-balanced
kNN distances, rankings, and gains/harms of alternative predictions. All features
are L2 normalized. Distance arithmetic float64; default k=11. Ranking ties follow
PyTorch argsort; discrete-vote ties are not evidence of confident alternatives.
The union Top-2 coverage uses up to four labels and must not be compared as an
equal-size candidate set. Gain/harm figures are descriptive, not an oracle rule.

Scope is ALL target train samples with one deterministic view, NOT the historical
low-confidence subset. To inspect old CSV IDs, join candidates by sample_id only
when file ordering/data are identical; checkpoint differences still change evidence.

Separate worktree preserves running jobs. Wait for GPU 0 if occupied. Default
batch_size=64, query_chunk=64, workers=4; lower batch/query chunk if memory is tight.
Source medoids are recomputed exactly and may take time. Missing class centers or
invalid target features fail rather than silently assign labels. Summary is printed
and saved automatically; rerun summarize_candidate_scan.py on the CSV offline.

Validation: syntax compilation and synthetic CSV gain/harm/coverage checks.
PyTorch/CUDA not available in authoring environment; real model scan pending.
