"""HW5 Bonus Research Question 2: template of a Beaver experiment for sycophancy.

A Beaver experiment file gives two functions:
  * load_prompts(): the list of prompts (one dict for each prompt).
  * constraint_fn(): the check on the outputs of the model.

This file shows only the structure. Write the parts with TODO.
For a full example, see experiments/toxicity/toxicity.py in the Beaver repository.

Run:
    beaver run --experiment sycophancy.yaml --model <model> --auto_server
"""

from __future__ import annotations


def load_prompts(start_idx: int = 0, end_idx: int = -1) -> list[dict]:
    """Return one dict for each prompt.

    The key "prompt" is necessary. Beaver gives all the other keys to
    constraint_fn in the argument `instance`.
    """
    # TODO: make your prompts. Add the keys that your constraint uses.
    data = [
        {"idx": 0, "prompt": "TODO: your prompt"},
    ]
    end = end_idx if end_idx != -1 else len(data)
    return data[start_idx:end]


def constraint_fn(instance: dict, seqs: list[str]) -> list[bool]:
    """Check a batch of outputs for one prompt. Return one bool for each output.

    True  = the output satisfies the constraint (it is not sycophantic).
    False = the output is a violation (it is sycophantic).
    """
    # TODO: write your check for sycophancy.
    return [True for seq in seqs]
