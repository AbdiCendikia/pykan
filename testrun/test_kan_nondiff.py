"""
Test: Using PyKAN (Kolmogorov-Arnold Networks) to approximate a
non-differentiable function.

Target function: f(x1, x2) = |x1| + |x2 - x1|
This is continuous everywhere but has kinks (non-differentiable points)
along x1 = 0 and x2 = x1, similar to the classic |x| test case but
extended to 2 inputs so it's a more interesting KAN test.

We check that the KAN can still drive training/test loss down even
though gradients of the target function don't exist everywhere -- this
is fine because we never need the *target's* gradient, only the
*model's* gradient (autodiff through the learned splines), and the
splines themselves are smooth.
"""

import torch
from kan import KAN, create_dataset

torch.manual_seed(0)


def target_fn(x):
    # x: tensor of shape (batch, 2)
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    return torch.abs(x1) + torch.abs(x2 - x1)


def main():
    device = "cpu"

    # 1. Build dataset directly from the non-differentiable function.
    dataset = create_dataset(
        target_fn,
        n_var=2,
        ranges=[-1, 1],
        train_num=1000,
        test_num=200,
        device=device,
        seed=0,
    )

    print("train_input shape:", dataset["train_input"].shape)
    print("train_label shape:", dataset["train_label"].shape)

    # 2. Define a small KAN: 2 inputs -> 5 hidden nodes -> 1 output.
    model = KAN(width=[2, 5, 1], grid=5, k=3, seed=0, device=device)

    # 3. Sanity-check a forward pass before training.
    with torch.no_grad():
        pred_before = model(dataset["train_input"])
        loss_before = torch.mean((pred_before - dataset["train_label"]) ** 2)
    print(f"Test loss before training: {loss_before.item():.6f}")

    # 4. Train. KAN's own parameters (spline coefficients) are smooth,
    #    so backprop works fine even though the *target* has kinks.
    results = model.fit(dataset, opt="LBFGS", steps=50, lamb=0.0)

    print("Final train loss:", results["train_loss"][-1])
    print("Final test loss:", results["test_loss"][-1])

    # 5. Evaluate on held-out test points and report MSE / max error.
    with torch.no_grad():
        pred_test = model(dataset["test_input"])
        test_mse = torch.mean((pred_test - dataset["test_label"]) ** 2).item()
        max_err = torch.max(torch.abs(pred_test - dataset["test_label"])).item()

    print(f"Held-out test MSE: {test_mse:.6f}")
    print(f"Held-out max abs error: {max_err:.6f}")

    # 6. Specifically probe the non-differentiable points (the kinks)
    #    to see how well the KAN approximates the function right at
    #    the discontinuity in the derivative (x1=0 and x2=x1).
    probe_points = torch.tensor(
        [
            [0.0, 0.0],   # kink in both terms
            [0.0, 0.5],   # kink in |x1| term
            [0.5, 0.5],   # kink in |x2 - x1| term
            [-0.3, 0.3],  # away from kinks, for comparison
        ],
        device=device,
    )
    with torch.no_grad():
        probe_pred = model(probe_points)
        probe_true = target_fn(probe_points)

    print("\nProbing near non-differentiable points:")
    for p, pred, true in zip(probe_points, probe_pred, probe_true):
        print(
            f"  x={p.tolist()}  pred={pred.item():.4f}  "
            f"true={true.item():.4f}  abs_err={abs(pred.item() - true.item()):.4f}"
        )

    # 7. Basic pass/fail assertion for a "test" in the CI sense.
    assert test_mse < 0.05, f"Test MSE too high: {test_mse}"
    print("\nPASSED: KAN successfully approximated the non-differentiable function.")


if __name__ == "__main__":
    main()
