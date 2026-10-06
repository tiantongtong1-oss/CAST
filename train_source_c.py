"""v8 full-source C entry: same integrated source/target flow as train.py.
Default: 30 source epochs + 30 target epochs. Use --source_only for source only.
"""
from train import run_training

if __name__ == '__main__':
    run_training()
