# Source class balancing v20

Base: 12c76107d6146d5d7763550674e5f9252934f530 (v19).
Branch: experiment/source-class-balance-fixed-v20.

Enable --source_class_weighting sqrt_inverse to derive weights from source_train.label only:
w[c] = sqrt(N / (C * count[c])), then normalize their arithmetic mean to 1.
Default none preserves unweighted behavior. Missing source classes fail explicitly.
For RAF counts the weights are approximately [0.889, 1.904, 1.192, 0.462, 0.717, 1.202, 0.635].

Pretraining CE is sum(weighted source CE) / sum(source sample weights).
During adaptation CE is (sum(weighted source CE) + sum(accepted unweighted target CE)) /
(sum(source sample weights) + accepted target count).
Unit weights reproduce original normalization. Evaluation CE stays unweighted.
No target class counts/labels are used to construct weights. Counts/weights are logged;
weighting mode is saved in checkpoint args. Affinity, prototype loss and rescue rules stay unchanged.

Also fixes a v19 source-pretraining crash caused by printing target-only counters before initialization.

Train a new source checkpoint: omit --checkpoint, use --pre_epochs 30.
Use --epochs 30 --support_threshold_mode fixed --support_fixed_threshold .92 to adapt in the same run.
Keep --immediate_prototype_update to match the previous experiment.
Run name must differ from historical runs. This starts from the normal backbone initialization,
not from the historical source checkpoint. No guarantee of improved Fear recall.
For a clean control rerun identical settings with --source_class_weighting none and a distinct run_name.
Existing behavior selects source checkpoints by target validation accuracy; unchanged here.
The combined run carries source optimizer/scheduler into adaptation, as before. To compare source
initializations under the historical fresh target optimizer, run --epochs 0 first, then use its
*_source_best.pth with --pre_epochs 0 --epochs 30 and a new run_name.

Validation: five new tests pass, including execution of actual adaptation loss expressions with
NumPy tensors, source-only weighting and source-loop regression. Syntax checks pass.
PyTorch/CUDA and RAF/FER training are unavailable in the authoring environment.

```bash
python -m unittest discover -s tests -p 'test_source_class_weights.py' -v
```
