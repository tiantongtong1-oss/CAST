# Global dual-view confidence quantile

Based on experiment/stable-rescue-competition-v24 at e2b617843b382df11b2f5117548d905262faf323.

At the beginning of each target epoch, the EMA teacher scans the entire target
training split using two independently sampled weak augmentations per image.
Only predictions agreeing across both views contribute. The score is the minimum
of the two maximum softmax probabilities. All classes and batches are pooled:

`tau = torch.quantile(scores, q, interpolation='higher')`

The default q is 0.70. There is one global scalar per epoch, shared by all classes.
The selector receives repeated copies solely to preserve its existing vector API.
Direct acceptance requires agreement and minimum confidence >= tau.
No confidence margin or lower/upper clipping is applied. If there are no valid
agreeing scores, --threshold_base (default 0.85) is the fallback.
Target ground-truth labels are never read by the scanner.

## Running

Append these options to your existing training command:

```bash
--threshold_quantile 0.70 --run_name global_quantile_higher_v25
```

The default run name is global_quantile_higher_v25 to keep output checkpoints
separate from the v24 defaults. Other source-radius, KNN, competition, stable
promotion, and loss settings retain the base branch behavior.

Legacy --threshold_beta, --threshold_margin, --threshold_min and --threshold_max
are accepted for CLI compatibility but do not affect this quantile threshold.
The epoch log reports q, tau, valid agreement count, scan acceptance fraction and
whether the fallback was used.

The nominal top 30% refers only to agreeing samples in the scan. Ties at the
inclusive boundary may increase that fraction; fresh augmentations and teacher
updates mean the training fraction need not match it. Rescue adds its own samples.
This scan now requires two teacher forward passes per batch instead of one.

## Validation

```bash
python -m unittest discover -s tests -p test_global_quantile_higher.py -v
```

These tests execute the actual scanner and selector function bodies with a NumPy
tensor shim. They cover pooled unequal batches, higher interpolation, both-view
minimum scores, disagreement, multiple predicted classes, fallback, finite-value
filtering, no clipping or history carryover, inclusive ties and invalid inputs.
They do not exercise real PyTorch kernels or full FER training.
