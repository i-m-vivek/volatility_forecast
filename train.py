# do random seeding
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset import StockData
import argparse
from src.model import VStock
import pickle
import torch.optim as optim
import copy
import time
import os
import random

start_time = time.strftime("%Y%m%d-%H%M%S")
print(start_time)
parser = argparse.ArgumentParser(description="Stock Volility Prediction Model")

parser.add_argument(
    "--max_len",
    default=300,
    type=int,
    help="Max number of sentences & audio files to use for prediction (default: 300)",
)
parser.add_argument(
    "--lr",
    default=0.0003,
    type=float,
    help="Learning rate to use for training (default: 0.0003)",
)
parser.add_argument(
    "--num_epochs",
    default=50,
    type=int,
    help="Number of epochs to run for training (default: 50)",
)
parser.add_argument(
    "--seed",
    default=2020,
    type=int,
    help="Seed for experiment (default: 2020)",
)
parser.add_argument(
    "--decay",
    default=1e-5,
    type=float,
    help="Weight decay to use for training (default: 1e-5)",
)
parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="Batch Size use for training the model (default: 32)",
)
parser.add_argument(
    "--test",
    action="store_true",
    default=False,
    help="Do testing at the end (default: False)",
)
parser.add_argument(
    "--use_stock_data",
    action="store_true",
    default=False,
    help="Whether to use previous available stock information or not (default: False)",
)


args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
# not doing seeding on CUDA
device = torch.device("cuda")


with open("data/sentence_embed.pkl", "rb") as f:
    embed = pickle.load(f)
traindata = StockData(
    "data/train.csv", embed, "data/MAEC_Dataset", args.max_len, 768, 29
)
valdata = StockData("data/val.csv", embed, "data/MAEC_Dataset", args.max_len, 768, 29)


trainloader = torch.utils.data.DataLoader(
    traindata, batch_size=args.batch_size, shuffle=True, num_workers=8
)
valloader = torch.utils.data.DataLoader(
    valdata, batch_size=args.batch_size, shuffle=True, num_workers=8
)

dataloaders = {"train": trainloader, "val": valloader}

model = VStock(768, 29, 256, 128, args.use_stock_data)
model.to(device).to(float)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)


if os.path.exists("saved_models") == False:
    os.mkdir("saved_models")
val_loss_history = []
train_loss_history = []

since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 9999999
best_epoch = 0
for epoch in range(args.num_epochs):
    print("Epoch {}/{}".format(epoch, args.num_epochs - 1))
    print("-" * 10)

    # Each epoch has a training and validation phase
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0

        # Iterate over data.
        for stock_data, target, sentence, audio, seq_len in dataloaders[phase]:
            stock_data = stock_data.to(device).to(float)
            target = target.to(device).to(float)
            sentence = sentence.to(device).to(float)
            audio = audio.to(device).to(float)
            seq_len = seq_len

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(stock_data, sentence, audio, seq_len)
                loss = criterion(outputs, target)

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * target.size(0)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)

        # deep copy the model
        if phase == "train":
            train_loss_history.append(epoch_loss)

        if phase == "val" and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if phase == "val":
            val_loss_history.append(epoch_loss)
            torch.save(
                {
                    "model_wts": model.state_dict(),
                    "current_epoch": epoch,
                    "best_model_wts": best_model_wts,
                    "best_epoch": best_epoch,
                    "best_loss": best_loss,
                    "val_loss_history": val_loss_history,
                    "train_loss_history": train_loss_history,
                    "args": args,
                },
                "saved_models/" + start_time + ".pth",
            )
        print(
            "{} Epoch: {} Loss: {:.4f} Best Val Loss: {:.4f}".format(
                phase, epoch, epoch_loss, best_loss
            )
        )
    print()

time_elapsed = time.time() - since
print(
    "Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
)
print("Best val Loss: {:4f}".format(best_loss))


model.load_state_dict(best_model_wts)
model.eval()

running_loss = 0.0
if args.test:
    testdata = StockData(
        "data/test.csv", embed, "data/MAEC_Dataset", args.max_len, 768, 29
    )
    testloader = torch.utils.data.DataLoader(
        testdata, batch_size=args.batch_size, shuffle=True, num_workers=8
    )

    for stock_data, target, sentence, audio, seq_len in testloader:
        stock_data = stock_data.to(device).to(float)
        target = target.to(device).to(float)
        sentence = sentence.to(device).to(float)
        audio = audio.to(device).to(float)
        seq_len = seq_len

        # forward
        with torch.no_grad():
            outputs = model(stock_data, sentence, audio, seq_len)
            loss = criterion(outputs, target)

        # statistics
        running_loss += loss.item() * target.size(0)

    test_loss = running_loss / len(testloader.dataset)
    print("-" * 25)
    print("Testing Loss: ", test_loss)

print("=" * 20)
print(start_time)
print(args)
