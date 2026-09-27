# Problem 1: targeted adversarial attacks

`fgsm.py` implements single-step targeted FGSM, `iterative_fgsm.py` repeats
projected updates, and `pgd.py` adds random restarts. Targets are 0 or 1;
the model and input use seed 13, with an L-infinity budget of 0.5.

Install uv first through https://docs.astral.sh/uv/#highlights. From this directory, then install dependencies (Python 3.12+ and uv required):

```bash
uv sync --locked
```

Run the experiment scripts with the desired target:

```bash
./run_experiments.sh 0
./run_experiments.sh 1
./run_pgd_experiments.sh 0
./run_pgd_experiments.sh 1
```

- `run_experiments.sh`: five single-step FGSM trials with the same fixed seed;
  saves `results_t0.txt` or `results_t1.txt`.
- `run_pgd_experiments.sh`: ten PGD trials with attack seeds 0–9, 100 random
  starts per trial, 100 steps per start, and step size 0.01;
  saves `results_pgd_t0.txt` or `results_pgd_t1.txt`.

Both scripts work from any working directory, overwrite their result file,
and record commands, stdout, stderr, and exit codes. They continue after failed
trials and exit with 0 if all pass, 1 if any fail, or 2 for invalid arguments.
