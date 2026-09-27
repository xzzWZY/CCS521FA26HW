"""Targeted L-infinity PGD with random starts and multiple restarts."""

import argparse
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PGDResult:
    adv_x: torch.Tensor
    success: bool
    iterations: int
    restarts_used: int


def pgd(model, x, target_class, eps=0.5, alpha=0.01, steps=100,
        restarts=100, seed=0):
    original = x.detach().clone()
    targets = torch.tensor([target_class], dtype=torch.long, device=x.device)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    budget = max(0.0, eps - 1e-7)
    adv_x = original.clone()
    iterations = 0
    restarts_used = 0

    for restart in range(restarts):
        iterations = 0
        restarts_used = restart
        noise = torch.empty_like(original).uniform_(-budget, budget, generator=generator)
        adv_x = (original + noise).detach()

        for _ in range(steps):
            adv_x.requires_grad_(True)
            with torch.enable_grad():
                logits = model(adv_x)
                if not 0 <= target_class < logits.shape[1]:
                    raise ValueError("target_class is outside the model's class range")
                loss = F.cross_entropy(logits, targets)
                gradient, = torch.autograd.grad(loss, adv_x)

            with torch.no_grad():
                candidate = adv_x - alpha * gradient.sign()
                delta = (candidate - original).clamp(-budget, budget)
                adv_x = (original + delta).detach()
                iterations += 1

                # Check success after every update, including the final one.
                logits = model(adv_x)
                if logits.argmax(dim=1).item() == target_class:
                    return PGDResult(adv_x, True, iterations, restart)

    return PGDResult(adv_x, False, iterations, restarts_used)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-class", "--t", type=int, choices=(0, 1), default=0)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="random-start seed")
    args = parser.parse_args()

    torch.manual_seed(13)
    model = nn.Sequential(
        nn.Linear(10, 10, bias=False), nn.ReLU(),
        nn.Linear(10, 10, bias=False), nn.ReLU(),
        nn.Linear(10, 3, bias=False),
    ).eval()
    x = torch.rand((1, 10))
    try:
        result = pgd(model, x, args.target_class, args.eps, args.alpha,
                     args.steps, args.restarts, args.seed)
    except ValueError as error:
        parser.error(str(error))

    with torch.no_grad():
        original_class = model(x).argmax(dim=1).item()
        new_class = model(result.adv_x).argmax(dim=1).item()
        distance = (result.adv_x - x).abs().max().item()
    print(f"Target Class: {args.target_class}")
    print(f"Model/input seed: 13; Attack seed: {args.seed}")
    print(f"Epsilon: {args.eps}; Alpha: {args.alpha}; Steps per restart: {args.steps}; Restarts: {args.restarts}")
    print(f"Iterations: {result.iterations}; Restarts used: {result.restarts_used}")
    print(f"Original Class: {original_class}")
    print(f"New Class: {new_class}")
    print(f"L-infinity perturbation: {distance:.9f}")
    print(f"Attack success: {result.success}")
    assert original_class == 2
    assert distance <= args.eps, "Perturbation exceeds epsilon"
    raise SystemExit(0 if result.success else 1)


if __name__ == "__main__":
    main()
