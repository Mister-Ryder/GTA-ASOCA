"""Independent author-formula and empty-patch mixed-precision regression checks.

Run CPU tests without occupying the training GPU:
  CUDA_VISIBLE_DEVICES='' python code/test_official_cldice.py --device cpu
Run the same half-output backward stress checks on CUDA after the queue permits:
  python code/test_official_cldice.py --device cuda

Author source pinned at 47d31a6cc4a8101b1ffe8052994821961e57af9f:
https://github.com/jocpae/clDice/blob/47d31a6cc4a8101b1ffe8052994821961e57af9f/cldice_loss/pytorch/cldice.py
The reference reduction uses dot products and vector sums independently of the
candidate loss. The unchanged soft skeleton is also checked against an
independent rendition of the author's ten-iteration min/max pooling sequence.
"""
import argparse
import json

import torch
import torch.nn.functional as F

from losses import cldice_loss, objective, soft_skeleton


def author_skeleton(image):
    def erode(value):
        a = -F.max_pool3d(-value, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        b = -F.max_pool3d(-value, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        c = -F.max_pool3d(-value, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(a, b), c)
    def opened(value):
        return F.max_pool3d(erode(value), (3, 3, 3), (1, 1, 1), (1, 1, 1))
    result = F.relu(image - opened(image))
    for _ in range(10):
        image = erode(image)
        delta = F.relu(image - opened(image))
        result = result + F.relu(delta - result * delta)
    return result


def author_reference(logits, labels):
    p = logits.float().softmax(1)[:, 1:2]
    t = labels[:, None].float()
    ps = author_skeleton(p).reshape(-1)
    ts = author_skeleton(t).reshape(-1)
    precision = (torch.dot(ps, t.reshape(-1)) + 1.) / (ps.sum() + 1.)
    sensitivity = (torch.dot(ts, p.reshape(-1)) + 1.) / (ts.sum() + 1.)
    return 1. - 2. * precision * sensitivity / (precision + sensitivity)


def old_cldice(logits, labels):
    p = logits.float().softmax(1)[:, 1:2]
    t = labels[:, None].float()
    ps, ts = soft_skeleton(p), soft_skeleton(t)
    axes = (1, 2, 3, 4)
    precision = ((ps * t).sum(axes) + 1e-5) / (ps.sum(axes) + 1e-5)
    sensitivity = ((ts * p).sum(axes) + 1e-5) / (ts.sum(axes) + 1e-5)
    return (1. - 2. * precision * sensitivity / (precision + sensitivity + 1e-5)).mean()


def run_tests(device="cpu"):
    torch.manual_seed(1419)
    device = torch.device(device)
    torch.set_num_threads(4)
    size = 16
    filled = torch.zeros(2, size, size, size, dtype=torch.long, device=device)
    filled[:, 2:-2, 7:10, 7:10] = 1
    empty = torch.zeros_like(filled)
    mixed = filled.clone()
    mixed[0] = 0
    rows = []
    for name, labels in (("filled", filled), ("empty", empty), ("mixed", mixed)):
        logits = torch.randn(2, 2, size, size, size, device=device, requires_grad=True)
        probability = logits.float().softmax(1)[:, 1:2]
        torch.testing.assert_close(soft_skeleton(probability), author_skeleton(probability), atol=0, rtol=0)
        expected = author_reference(logits, labels)
        actual = cldice_loss(logits, labels)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        assert 0 <= actual <= 1 and torch.isfinite(actual)
        actual.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        rows.append({"case": name, "loss": float(actual.detach()),
                     "gradient_max": float(logits.grad.abs().max())})

    # Perfect foreground-only matches and wholly empty skeletons have zero
    # author loss; no added harmonic epsilon produces a nonzero empty floor.
    logits_empty = torch.zeros(2, 2, size, size, size, device=device)
    torch.testing.assert_close(cldice_loss(logits_empty, empty), torch.zeros((), device=device), atol=0, rtol=0)
    logits_perfect = torch.stack([torch.where(filled.bool(), -40., 40.),
                                 torch.where(filled.bool(), 40., -40.)], dim=1)
    torch.testing.assert_close(cldice_loss(logits_perfect, filled), torch.zeros((), device=device), atol=1e-6, rtol=0)

    # A near-uniform ambiguous empty patch has a tiny positive soft skeleton.
    # Half logits can represent this around zero. The old tiny-smoothing loss
    # produces finite forward values but overflows half-output backward at the
    # real fixed scale1024; smooth1 keeps that exact stress case finite.
    template = torch.zeros(2, 2, size, size, size, device=device, dtype=torch.float16)
    template[:, 1, size // 2, size // 2, size // 2] = 4e-5
    stress = []
    for name, function in (("old_eps1e-5", old_cldice), ("author_smooth1", cldice_loss)):
        logits = template.clone().requires_grad_(True)
        loss = function(logits, empty)
        (1024 * .2 * loss).backward()
        finite = bool(torch.isfinite(logits.grad).all())
        stress.append({"implementation": name, "loss": float(loss.detach()),
                       "scaled_half_output_gradient_finite": finite,
                       "scaled_gradient_max": float(logits.grad.abs().max()) if finite else None})
        if name == "old_eps1e-5":
            assert torch.isfinite(loss) and not finite, "Old regression case did not reproduce half-gradient overflow"
        else:
            assert finite and logits.grad.abs().sum() > 0

    # Verify the topology term remains additive0.2 and the same segmentation
    # objective/DS outputs are used, rather than silently replacing the base.
    logits = torch.randn(2, 2, size, size, size, device=device, requires_grad=True)
    output = {"logits": logits, "aux_logits": [], "classifier": {}, "gaussian": None}
    batch = {"label": mixed, "priors": torch.zeros(2, 3, dtype=torch.long, device=device)}
    base, _ = objective(output, batch, "unet")
    combined, _ = objective(output, batch, "unet_cldice")
    torch.testing.assert_close(combined - base, .2 * cldice_loss(logits, mixed), atol=1e-6, rtol=1e-5)
    combined.backward()
    assert torch.isfinite(logits.grad).all()
    return {"passed": True, "device": str(device), "known_cases": rows,
            "half_output_regression": stress, "base_objective_unchanged": True,
            "amp_retry_added": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    print(json.dumps(run_tests(args.device), allow_nan=False, indent=2))
