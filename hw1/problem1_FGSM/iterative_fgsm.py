"""Targeted iterative FGSM with projection onto an L-infinity ball."""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


def iterative_fgsm(
    model, x, target_class, eps=0.5, alpha=0.01, steps=None,
    *, return_iterations=False,
):
    """Attack one input with a leading batch dimension of size 1."""
    if x.ndim < 2 or x.shape[0] != 1:
        raise ValueError("x must contain exactly one sample (batch size 1)")
    original = x.detach().clone()
    adv_x = original.clone()
    targets = torch.tensor([target_class], dtype=torch.long, device=x.device)
    # Leave a small margin for floating-point rounding, as in fgsm.py.
    budget = max(0.0, eps - 1e-7)
    iteration = 0
    while True:
        adv_x.requires_grad_(True)
        with torch.enable_grad():
            logits = model(adv_x)
            if logits.argmax(dim=1).item() == target_class:
                result = adv_x.detach()
                return (result, iteration) if return_iterations else result
            if steps is not None and iteration >= steps:
                raise RuntimeError(f"Attack did not succeed within {steps} steps")
            loss = F.cross_entropy(logits, targets)
            gradient, = torch.autograd.grad(loss, adv_x)
        with torch.no_grad():
            # Subtract the gradient sign to move toward the target class.
            candidate = adv_x - alpha * gradient.sign()
            # Apply the clip
            delta = (candidate - original).clamp(-budget, budget)
            candidate = (original + delta).detach()
            if torch.equal(candidate, adv_x):
                raise RuntimeError(
                    f"Attack stalled after {iteration} steps without reaching the target"
                )
            adv_x = candidate
            iteration += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--x", type=float, nargs=10, required=True, metavar="VALUE",
        help="the 10 feature values of a single input",
    )
    parser.add_argument("--target-class", "--t", type=int, choices=(0, 1), default=0)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument(
        "--max-steps", "--steps", dest="max_steps", type=int, default=None,
        help="optional iteration limit (default: iterate until success)",
    )
    args = parser.parse_args()

    torch.manual_seed(13)
    model = nn.Sequential(
        nn.Linear(10, 10, bias=False), nn.ReLU(),
        nn.Linear(10, 10, bias=False), nn.ReLU(),
        nn.Linear(10, 3, bias=False),
    )
    model.eval()
    x = torch.tensor([args.x], dtype=model[0].weight.dtype)
    try:
        adv_x, iterations = iterative_fgsm(
            model, x, args.target_class, args.eps, args.alpha, args.max_steps,
            return_iterations=True,
        )
    except ValueError as error:
        parser.error(str(error))
    except RuntimeError as error:
        parser.exit(1, f"Attack failed: {error}\n")

    with torch.no_grad():
        original_class = model(x).argmax(dim=1).item()
        new_class = model(adv_x).argmax(dim=1).item()
        distance = (adv_x - x).abs().max().item()
    print(f"Target Class: {args.target_class}")
    print(f"Seed: 13; Iterations: {iterations}; Alpha: {args.alpha}; Epsilon: {args.eps}")
    print(f"Original Class: {original_class}")
    print(f"New Class: {new_class}")
    print(f"L-infinity perturbation: {distance:.9f}")
    print(f"Attack success: {new_class == args.target_class}")
    assert distance <= args.eps, "Perturbation exceeds epsilon"
    assert new_class == args.target_class, "Attack did not reach the target class"


if __name__ == "__main__":
    main()
