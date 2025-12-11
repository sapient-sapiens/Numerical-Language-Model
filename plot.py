import argparse 
import os 

import matplotlib.pyplot as plt 
import pandas as pd 

def load_eval_log(run_dir): 
    path = os.path.join(run_dir, "eval_log.csv") 
    if not os.path.exists(path): 
        raise FileNotFoundError(f"eval_log.csv not found in {run_dir}") 
    df = pd.read_csv(path) 
    df = df.sort_values(["op", "num_digits", "step"]).reset_index(drop=True)
    return df

def plot_each_task(df, out_dir): 
    ops = ["*", "+"]
    digits = [1, 2, 3, 4, 5, 6]
    for op in ops:
        fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True, sharey=True)
        axes = axes.flatten()
        for i, digit in enumerate(digits):
            ax = axes[i]
            now = df[(df["op"] == op) & (df["num_digits"] == digit)]
            ax.plot(now["step"], now["accuracy"], marker="o")
            ax.set_title(f"{op} with {digit} digit(s)")
            ax.set_xlabel("Step")
            ax.set_ylabel("Accuracy")
            ax.grid(True, linestyle="--", alpha=0.6)
        fig.suptitle(f"Accuracy v.s. step for operator '{op}'")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(out_dir, f"accuracy_by_digits_{op}.png")
        plt.savefig(out_path)
        plt.close(fig)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--run-dir", type = str, required = True) 
    parser.add_argument("--output-dir", type = str, default = ".")
    args = parser.parse_args() 
    df = load_eval_log(args.run_dir) 
    out_dir = os.path.join(args.output_dir, "plots") 
    os.makedirs(out_dir, exist_ok=True) 
    plot_each_task(df, out_dir) 