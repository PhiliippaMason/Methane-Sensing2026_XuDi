from models import DeepUNet
from trainer import Trainer
from loader import MethaneLoaderNewQ
import sys
import torch
import torch.nn as nn

from torch.utils.data import DataLoader


def loss(pred, target):
    mse_loss = nn.MSELoss()
    ll = mse_loss(pred, target)
    # ll = ll.sum(dim=(-2,-1)) #*mask
    return ll


# python3 train.py 12 FINAL_12/

# Input arguments
channels = int(sys.argv[1])
out_dir = sys.argv[2]

# Set up
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda")

# Set up model
model = DeepUNet(in_channels=channels, out_channels=1)
model = model.to(device)
model = nn.DataParallel(model)

# Set up loss function
loss_fn = loss

train_dataset = MethaneLoaderNewQ(
    device="cuda", mode="train", plume_id=None, channels=channels
)
test_dataset = MethaneLoaderNewQ(
    device="cuda", mode="test", plume_id=None, channels=channels
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Make the trainer
trainer = Trainer(model, train_loader, test_loader, train_dataset, loss, out_dir, 1e-3)

# Train
trainer.train(n_epochs=150)
