"""Record real source covariance refreshes; plots summarize coordinate means."""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

class CovarianceHistory:
    def __init__(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.history = []
        self.csv_path = os.path.join(output_dir, "diagonal_covariance.csv")
        with open(self.csv_path, "w", newline="") as handle:
            csv.writer(handle).writerow(
                ["epoch", "refresh", "class_id", "class_name", "dimension",
                 "momentum", "previous_variance", "observed_variance",
                 "updated_variance", "delta"])
        self.plot_path = os.path.join(output_dir, "covariance_evolution.png")

    def record(self, bank, epoch):
        ready = bank.distribution_initialized.detach().cpu().numpy()
        previous = bank.previous_class_variances.detach().cpu().numpy()
        previous_ready = bank.previous_covariance_initialized.detach().cpu().numpy()
        observed = bank.observed_class_variances.detach().cpu().numpy()
        updated = bank.class_variances.detach().cpu().numpy()
        beta = float(bank.current_global_momentum.item())
        refresh = int(bank.refresh_count.item())
        summary = np.full((3, bank.num_classes), np.nan)
        with open(self.csv_path, "a", newline="") as handle:
            writer = csv.writer(handle)
            for c in np.flatnonzero(ready):
                name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
                # First estimate is initialized directly; previous placeholder is not an estimate.
                first = not previous_ready[c]
                before = np.full_like(previous[c], np.nan) if first else previous[c]
                delta = updated[c] - before
                summary[:, c] = [np.mean(before), observed[c].mean(), updated[c].mean()]
                print("[V9 covariance] epoch=%d class=%s beta=%.4f previous_mean=%.8g "
                      "observed_mean=%.8g updated_mean=%.8g mean_abs_delta=%.8g" %
                      (epoch, name, beta, summary[0,c], summary[1,c], summary[2,c],
                       np.mean(np.abs(delta))), flush=True)
                for d in range(bank.feature_dim):
                    writer.writerow([epoch, refresh, int(c), name, d, beta,
                                     float(before[d]), float(observed[c,d]),
                                     float(updated[c,d]), float(delta[d])])
        self.history.append((epoch, summary))
        fig, axes = plt.subplots(2, 1, figsize=(11, 9))
        epochs = [item[0] for item in self.history]
        values = np.stack([item[1] for item in self.history])
        for c in range(bank.num_classes):
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
            line, = axes[0].plot(epochs, values[:,2,c], marker=".", label=name)
            axes[1].plot(epochs, values[:,1,c], linestyle="--", alpha=.55,
                         color=line.get_color(), label=name + " observed")
            axes[1].plot(epochs, values[:,2,c], color=line.get_color(),
                         label=name + " EMA")
        axes[0].set_title("Per-class diagonal covariance: mean over feature dimensions")
        axes[1].set_title("Observed variance (dashed) and updated variance (solid)")
        for ax in axes:
            ax.set_xlabel("Target epoch (zero based)")
            ax.set_ylabel("Mean coordinate variance")
            ax.grid(alpha=.25)
            ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=160)
        plt.close(fig)
        print("[V9 covariance] CSV: %s Plot: %s" % (self.csv_path, self.plot_path), flush=True)
