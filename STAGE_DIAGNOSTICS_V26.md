# v25 read-only stage diagnostics (v26)

Base: v25 0c5872de8a3ad3f2866272cb3cbabe78e3cbd4dc.
Branch: experiment/v25-stage-diagnostics-v26. No filter, threshold, weight,
training role promotion, loss or optimizer changes.

Each target epoch reports training counts by predicted class, WITHOUT consuming
target training labels. teacher_raw is view-1 argmax; agreement is two-view class
agreement. high_confidence is a PARALLEL direct path, not a rescue funnel step.
Rescue funnel: rescue_candidate (agree & not high confidence), rescue_distance
(at least 3 close neighbors), rescue_vote (adaptive vote), rescue_probability
(both-view probability margin), rescue_prototype (source prototype advantage).
The funnel is cumulative. final_accept, supervised and anchors are separate
final outcomes. During rescue warmup the rescue audit stages are zero by design.

Validation diagnostics use a separate dual-weak FER val loader, zero workers,
private seeded generator and isolated Python/NumPy/Torch CPU/CUDA RNG context.
The seed is fixed (91314) each epoch to hold diagnostic augmentations steady.
There is an independent validation-only role tracker with val-local IDs; it never
writes training roles/prototypes or uses labels for decisions. True val labels
are consumed only by the stats accumulator AFTER all masks are decided.
Teacher/source-bank/threshold snapshot is taken at START of each epoch; Student
validation remains the existing END-of-epoch deterministic test-transform run.
Validation surrogate anchors are hypothetical across validation snapshots, not
actual training anchors. Actual training counts are collected across changing
within-epoch EMA states; do not expect val and train distributions to match exactly.

For class c at stage s:
precision = selected_correct_c / selected_predicted_as_c.
correct_coverage = selected_correct_c / ALL true validation class-c samples.
No selection or no true support is NaN (undefined), not zero/perfect precision.
Confusion rows=true, columns=predicted. Selected-stage matrices omit rejected
samples; use full true support in CSV to calculate coverage.

Outputs in models/rafdb_fer/ with the run_name prefix:
- _stage_diagnostics.csv: epoch,split,stage,class,selected,correct,true_support,
  precision,correct_coverage. Training correctness/quality columns are blank.
- _stage_diagnostics_confusion.csv: Teacher per-stage validation confusion rows.
Standard Student validation/test confusion matrices are also printed when existing
metrics are reported. Test is NOT used for stage quality diagnosis.

Use v25 training command unchanged except --run_name v25_stage_diagnostics_v26.
Continue using same SOURCE checkpoint and --pre_epochs 0; restart target adaptation
for history diagnostics. Loading an adapted checkpoint here is weight initialization,
not an exact resume, and cannot reconstruct prior anchor history. Running the full
30 target epochs gives comparable end results, but first epochs can reveal bottlenecks.
This adds two val forward passes per image and rescue KNN auditing each epoch;
no repeated source calibration is added by diagnostics.

Validation: 4 stats tests, 5 adaptive-neighbor tests, 6 competition tests pass;
syntax/diff checks pass. PyTorch, CUDA and data unavailable here; full runtime,
RNG trajectory equivalence and validation history integration are not executed.
