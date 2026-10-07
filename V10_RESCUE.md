# V10: v7 dual-view KNN rescue with quarantine

Based on experiment/source-global-gaussian-knn-reliability-v7.
The source Gaussian and shared kernel bandwidth remain unchanged.

## Acceptance
- Agreement + two confident predictions: original acceptance.
- Agreement + existing KNN rescue: low-weight rescue.
- Disagreement: query EACH view separately. Compare the two teacher-predicted
  candidate classes using weighted neighbor support. Both queries must choose
  the same class, pass the existing Gaussian/density reliability gate, and meet
  --knn_support_margin (default 0.2). No merged feature is used for this decision.
- Rescue starts at --knn_warmup_epochs as in v7.

## Isolation
All rescued IDs are quarantined persistently for the run (checkpointed).
Quarantined accepted samples use --rescue_weight (default 0.2) for classification.
They do not participate in category MMD, prototype consistency, or target
prototype updates, even if later accepted by the confidence path.
Their features remain in the target memory for geometric density, but their
labels cannot contribute neighbor support. Registry survives memory refresh.
There is no automatic promotion or AU auditor in this experiment.
Non-quarantined memory labels remain teacher predictions, NOT verified labels.
Quarantine does not stop indirect influence through student/teacher updates.

## Run
In the existing CUDA/PyTorch CAST environment:
```bash
python -m unittest discover -s tests -p 'test_dual_view_quarantine.py'
python train.py --data1 rafdb --data2 fer --backbone mobilenet_v2 \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --pre_epochs 30 --epochs 30 --rescue_weight 0.2 --knn_support_margin 0.2
```
For your already tested bandwidth, additionally use
--knn_bandwidth_multiplier 22.627417. This is an experimental setting, not
a verified improvement.
New checkpoint filenames contain v10. --checkpoint loads model weights only;
it is not full resume (same behavior as v7).

Logs add Disagreement_Rescue_Num and Quarantined_Total.
KNN_Rescue_Num now includes agreement and disagreement rescues; KNN_Pass_Num
and Mean_Reliability still describe the original agreement gate.
Prototype_Agreement now covers only non-quarantined confident samples.
Evaluate rescue precision, classwise confusion, and retention on independent data.
Shared features can still generate correlated errors; two-view KNN agreement
is not proof of correct semantics.
