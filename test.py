import torch
from src.model import VStock
from src.dataset import StockData
import argparse
import torch.nn as nn
import pickle


parser = argparse.ArgumentParser(description="Stock Volility Prediction Model")

parser.add_argument(
    "--file",
    type=str,
    help="<Required> Filename to test a saved model",
)
args = parser.parse_args()
device = torch.device("cuda")
file = torch.load("saved_models/" + args.file)
args = file["args"]
model = VStock(768, 29, 256, 128, args.use_stock_data)
model.to(device).to(float)

model.load_state_dict(file["best_model_wts"])
model.eval()


with open("data/sentence_embed.pkl", "rb") as f:
    embed = pickle.load(f)


running_loss = 0.0
criterion = nn.MSELoss()
testdata = StockData("data/test.csv", embed, "data/MAEC_Dataset", args.max_len, 768, 29)
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
print(args)
print("-" * 25)
