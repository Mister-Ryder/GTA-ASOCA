"""Historical GTAirway objective with matched segmentation/DS for baselines.

Explicit corrections/adaptations: background Dice uses weights in both numerator
and denominator (the historical numerator-only form can be negative); coronary
attribute focal losses use uniform alpha instead of airway-specific class
weights; classifier supervision starts jointly from scratch at epoch zero.
Stored unit Gaussian fields are converted to the historical .5+.5*field targets.
No-balancing removes both anatomy loss weights and the trainer's balanced sampler.
"""
import torch
import torch.nn.functional as F


def weighted_dice_loss(prediction, target, weights, power=None, smooth=.01):
    """Historical flattened squared-Dice, with a symmetrically weighted ratio.

    For power=2 the foreground transform is exactly (p+.1)^2, retaining the
    historical positive delta and its nonzero perfect-foreground loss floor.
    Nonnegative weights give 2*a*b <= a^2+b^2, so the loss is bounded in [0,1].
    """
    p, t, w = (value.float().reshape(-1) for value in (prediction, target, weights))
    if p.shape != t.shape or w.shape != t.shape:
        raise ValueError("Prediction, target and weights must contain equal voxel counts")
    if power is not None:
        p = (p + .1).pow(power)
    numerator = 2 * (p * t * w).sum() + smooth
    denominator = ((p.square() + t.square()) * w).sum() + smooth
    return (1 - numerator / denominator).clamp(0, 1)


def segmentation_loss(logits, labels, sample_weights=None):
    if logits.shape[1] != 2:
        raise ValueError("This faithful ASOCA objective expects background plus coronary lumen")
    if logits.shape[2:] != labels.shape[1:]:
        # Upsample logits; nearest target downsampling can delete thin vessels.
        logits = F.interpolate(logits, size=labels.shape[1:], mode="trilinear", align_corners=False)
    probability = logits.float().softmax(1)
    truth = labels[:, None].float()
    if sample_weights is None:
        sample_weights = torch.ones(labels.shape[0], device=labels.device)
    weights = sample_weights[:, None, None, None, None].expand_as(truth)
    background = weighted_dice_loss(probability[:, 0:1], 1 - truth, weights, smooth=.01)
    foreground = weighted_dice_loss(probability[:, 1:2], truth, weights, power=2, smooth=1)
    return background + foreground


def focal_loss(logits, labels, gamma=2):
    # Uniform alpha is deliberate: historical airway class weights do not map
    # to ASOCA geometric branch-depth, structure and location labels.
    ce = F.cross_entropy(logits.float(), labels, reduction="none")
    return ((1 - torch.exp(-ce)).pow(gamma) * ce).mean()


def soft_skeleton(x, iterations=10):
    def erode(t):
        return torch.minimum(torch.minimum(-F.max_pool3d(-t, (3,1,1), 1, (1,0,0)),
                                           -F.max_pool3d(-t, (1,3,1), 1, (0,1,0))),
                             -F.max_pool3d(-t, (1,1,3), 1, (0,0,1)))
    def opening(t):
        return F.max_pool3d(erode(t), 3, 1, 1)
    skeleton = F.relu(x - opening(x))
    for _ in range(iterations):
        x = erode(x)
        delta = F.relu(x - opening(x))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def cldice_loss(logits, labels):
    p = logits.float().softmax(1)[:, 1:2]
    truth = labels[:, None].float()
    psk, tsk = soft_skeleton(p), soft_skeleton(truth)
    dims = (1,2,3,4)
    precision = ((psk * truth).sum(dims) + 1e-5) / (psk.sum(dims) + 1e-5)
    sensitivity = ((tsk * p).sum(dims) + 1e-5) / (tsk.sum(dims) + 1e-5)
    return (1 - 2 * precision * sensitivity / (precision + sensitivity + 1e-5)).mean()


def objective(output, batch, method, epoch=0):
    target = batch["label"]
    priors = batch.get("priors")
    sample_weights = torch.ones(target.shape[0], device=target.device)
    if method.startswith("gta") and method != "gta_no_balance":
        generation, structure = priors[:, 0], priors[:, 1]
        gen_weights = torch.where(generation >= 5, 1.5,
                                 torch.where(generation >= 3, 1.2, 1.))
        sample_weights = gen_weights * (1 + .3 * (structure == 3).float())
    heads = [output["logits"]] + list(output.get("aux_logits", []))
    if len(heads) > 3:
        raise ValueError("Expected final logits plus at most two decoder supervision heads")
    weights = [1., .5, .25][:len(heads)]
    seg = sum(weight * segmentation_loss(head, target, sample_weights)
              for weight, head in zip(weights, heads)) / sum(weights)
    loss = seg
    details = {"segmentation": float(seg.detach())}
    cls = output.get("classifier", {})
    if cls:
        classification = (focal_loss(cls["generation"], priors[:, 0])
                          + 1.5 * focal_loss(cls["structure"], priors[:, 1])
                          + focal_loss(cls["location"], priors[:, 2]))
        uncertainty = cls.get("uncertainty_estimates", {})
        if "structure" in uncertainty:
            # Exact historical correctness target, not another attribute CE.
            correct = cls["structure"].detach().argmax(1).eq(priors[:, 1]).float()
            certainty = uncertainty["structure"].float().flatten(1).softmax(1)
            confidence = certainty.gather(1, priors[:, 1:2]).squeeze(1)
            certainty_loss = F.mse_loss(confidence, correct)
            classification = classification + .5 * certainty_loss
            details["structure_correctness"] = float(certainty_loss.detach())
        cls_weight = min(.5, .1 + max(0, epoch) * .01)
        loss = loss + cls_weight * classification
        details.update(classification=float(classification.detach()), classifier_weight=cls_weight)
        if "guidance_map" in cls:
            guidance = cls["guidance_map"].float()
            pooled = F.avg_pool3d(target[:, None].float(), kernel_size=4, stride=4).detach()
            if guidance.shape[2:] != pooled.shape[2:]:
                guidance = F.interpolate(guidance, size=pooled.shape[2:],
                                         mode="trilinear", align_corners=False)
            guide_loss = F.mse_loss(guidance, pooled.expand_as(guidance))
            loss = loss + .1 * guide_loss
            details["guidance_map"] = float(guide_loss.detach())
        # Historical enhanced_features are returned but unused by its loss;
        # retain this behavior rather than inventing a new feedback objective.
    gaussian = output.get("gaussian")
    if gaussian is not None:
        gaussian_target = .5 + .5 * batch["gaussian"].float()
        field = F.mse_loss(gaussian.float(), gaussian_target)
        loss = loss + 10 * field
        details["gaussian"] = float(field.detach())
    if method == "unet_cldice":
        topology = cldice_loss(output["logits"], target)
        loss = loss + .2 * topology
        details["cldice"] = float(topology.detach())
    return loss, details


def synthetic_loss_tests(device="cpu"):
    """Known ratios, historical delta floor, finite gradients and bounded loss."""
    device = torch.device(device)
    p = torch.tensor([0., 1., .25, .75], device=device)
    t = torch.tensor([0., 1., 0., 1.], device=device)
    w = torch.tensor([1., 1.95, 1.2, 1.5], device=device)
    for power, smooth in ((None, .01), (2, 1.)):
        actual = weighted_dice_loss(p, t, w, power=power, smooth=smooth)
        q = p if power is None else (p + .1).square()
        expected = 1 - (2 * (q * t * w).sum() + smooth) / (((q.square() + t.square()) * w).sum() + smooth)
        torch.testing.assert_close(actual, expected)
        assert 0 <= actual <= 1
    # Symmetric background Dice is exactly zero for a perfect mask even with
    # >1 anatomy weights; the old numerator-only ratio could be negative.
    perfect = weighted_dice_loss(t, t, w)
    torch.testing.assert_close(perfect, torch.zeros((), device=device))
    one = torch.ones(512, device=device)
    floor = weighted_dice_loss(one, one, one, power=2, smooth=1)
    expected_floor = 1 - (2 * 1.21 * 512 + 1) / ((1.21**2 + 1) * 512 + 1)
    torch.testing.assert_close(floor, torch.tensor(expected_floor, device=device), atol=1e-6, rtol=1e-5)
    torch.manual_seed(1419)
    for foreground_fraction in (0., .01, .5, 1.):
        labels = (torch.rand(2, 16, 16, 16, device=device) < foreground_fraction).long()
        logits = torch.randn(2, 2, 16, 16, 16, device=device, requires_grad=True)
        loss = segmentation_loss(logits, labels, torch.tensor([1., 1.95], device=device))
        assert torch.isfinite(loss) and 0 <= loss <= 2
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0
    return {"loss_tests": True, "perfect_squared_dice": float(perfect),
            "historical_perfect_foreground_power_dice_floor": float(floor)}


if __name__ == "__main__":
    print(synthetic_loss_tests())
