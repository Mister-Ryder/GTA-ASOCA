"""MONAI SwinUNETR baseline with existing decoder features supervised."""
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinBaseline(nn.Module):
    def __init__(self, deep_supervision=True):
        super().__init__()
        self.net = SwinUNETR(in_channels=1, out_channels=2, feature_size=24, use_checkpoint=True)
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.head_half = nn.Conv3d(24, 2, 1)
            self.head_quarter = nn.Conv3d(48, 2, 1)
            self.features = {}
            self.net.decoder2.register_forward_hook(self._capture("half"))
            self.net.decoder3.register_forward_hook(self._capture("quarter"))

    def _capture(self, key):
        def hook(module, inputs, output):
            self.features[key] = output
        return hook

    def forward(self, image):
        logits = self.net(image)
        aux = []
        if self.deep_supervision:
            aux = [self.head_half(self.features.pop("half")), self.head_quarter(self.features.pop("quarter"))]
        return {"logits": logits, "aux_logits": aux, "classifier": {}, "gaussian": None}
