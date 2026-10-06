# Experiment C: multiple source prototypes, same hinge margin

Branch: experiment/source-multi-prototype-margin-C
Based on B. train_source_c.py uses the same source split (80/20, seed 2000),
training losses/weights, augmentation, optimizer, initialization and selection.
The only method change is one versus K prototypes per source true class.
Defaults: K=3, 20 spherical k-means iterations, margin=0.1, weight=0.1.
The positive score is the maximum cosine over true-class prototypes; negative
score is maximum cosine over every other-class prototype. Mean hinge is unchanged.

At each epoch start, source training IDs only are embedded using fixed views.
Features are L2-normalized and clustered separately per genuine class on CPU.
Initialization is deterministic: nearest member to class mean then farthest members.
Empty clusters are reseeded to poorly represented members; printed cluster counts
may include empty modes for degenerate features. Centers remain detached for the
whole epoch. No source validation/test or target labels enter prototype estimation.
K=1 reproduces the B prototype definition (small numerical differences possible).
No target gate, multi-prototype target bank or additional target loss is introduced.

## Run

```bash
python -u train_source_c.py --epochs 30 --prototypes_per_class 3 --cluster_iterations 20 --margin_weight 0.1 --margin 0.1 --run_name source_C > source_C.log 2>&1
```
Do not load A/B weights for the clean comparison. Optional --checkpoint is fine-
tuning only, with the inherited validation previously-seen warning. The existing
B script remains available. Keep all other settings matched across A/B/C.

Output: new_models/rafdb_fer/source_C/source_C_best_macro_f1.pth and split.json.
Validation macro-F1 selects the checkpoint; all class precision/recall/F1 are
printed. RAFDB test is evaluated only after selection. Existing output folders
are rejected. Margin prototypes and args are saved inside the checkpoint.

## FER audit (no training)

```bash
python -u train.py --pre_epochs 0 --checkpoint new_models/rafdb_fer/source_C/source_C_best_macro_f1.pth --audit_only --run_name fer_audit_C > fer_audit_C.log 2>&1
```
Compare FER validation macro-F1 and disgust precision/recall/F1 to A/B. No gain
is guaranteed; do not repeatedly optimize on RAFDB test results. If proceeding
to target training preserve the original knn_bandwidth_multiplier=22.627417.

Checks: python -m unittest discover -s tests; python -m py_compile train_source_c.py
multi_prototypes.py multi_source_margin.py. Local environment has no PyTorch/CUDA;
provided gradient/equivalence tests must run on the training server.
