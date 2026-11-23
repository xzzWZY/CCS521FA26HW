# HW5 Bonus Research Question 2: sycophancy with Beaver

Template of a Beaver experiment. It shows the structure only. You design the prompts and the check for sycophancy.

- `sycophancy.py`: the prompts (`load_prompts`) and the constraint (`constraint_fn`).
- `sycophancy.yaml`: the settings of the run.

Beaver code: https://github.com/BeaverFramework/Beaver. Read its README first. The toxicity experiment (`experiments/toxicity/toxicity.py`, `toxicity.yaml` and `toxicity_baseline.yaml`) is a full example.

```bash
beaver run --experiment sycophancy.yaml --model <model> --auto_server
```
