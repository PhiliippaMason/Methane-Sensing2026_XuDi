from src.model import UNet
import torch


def test_unet_forward_shape():
    m = UNet(in_channels=3, out_channels=1)
    x = torch.rand(1, 3, 128, 128)
    y = m(x)
    assert y.shape == (1, 1, 128, 128)
