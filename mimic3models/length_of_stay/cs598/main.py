import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from preprocess import preprocess
from model import PhysioNet
from model import EpisodeNet
from model import NotesNet
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from tqdm import trange
import time
import argparse
import statistics
import matplotlib
import matplotlib.pyplot as plt
from math import log
from datetime import datetime
from functools import lru_cache
import gc

seed = 29
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
NOTES_PATH = "/mnt/data01/mimic-3/benchmark-notes/"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--lr", type=float, default=0.001, help="Learning rate for the model"
)
parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--window", type=int, default=5, help="minimum 5")
parser.add_argument("--use_saved", type=bool, default=True, help="use saved embeddings")
parser.add_argument("--data", type=str, help="sample / all", default="all")
parser.add_argument(
    "--train_sample_size",
    type=int,
    help="if data == sample, provide training sample size",
    default=5000,
)
parser.add_argument(
    "--val_sample_size",
    type=int,
    help="if data == sample, provide training sample size",
    default=500,
)
parser.add_argument(
    "--test_sample_size",
    type=int,
    help="if data == sample, provide training sample size",
    default=5000,
)
parser.add_argument(
    "--model_output_dir",
    type=str,
    help="Model output dir",
    default="/mnt/data01/models/cnn/",
)
parser.add_argument("--mode", type=str, help="physio / model / both", default="both")

args = parser.parse_args()
print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=200)
def load_notes_embeddings(patient_id, episode, folder="train"):
    file = "{}{}/{}/{}_notes_bert_window5-1_tensor.parquet".format(
        NOTES_PATH, folder, patient_id, episode
    )
    df = pq.read_pandas(file).to_pandas()
    return torch.tensor(
        np.stack([np.stack(x) for x in df["TEXT_WINDOW_EMBEDDING"].iloc[0]])
    )


def eval_model(model, val_loader):
    model.eval()
    # all_y_true = torch.DoubleTensor()
    # all_y_pred = torch.DoubleTensor()
    mses = []
    for x, y in val_loader:
        # x = x.to(device)
        x = x.cuda()
        # print("x_val is {}".format(x.is_cuda))
        y_hat = model(x)
        mse = mean_squared_error(y.detach().numpy(), y_hat.cpu().detach().numpy())
        mses.append(mse)
        # all_y_true = torch.cat((all_y_true, y), dim=0)
        # all_y_pred = torch.cat((all_y_pred,  y_hat.to('cpu')), dim=0)
    mse = statistics.mean(mses)
    # mse= mean_squared_error(all_y_true.detach().numpy(), all_y_pred.detach().numpy())
    print(f"mse: {mse:.3f}")
    return mse


def train(model, train_loader, val_loader, n_epochs, optimizer, criterion):
    train_losses = [0 for i in range(n_epochs)]
    mses = [0 for i in range(n_epochs)]
    for epoch in trange(n_epochs):
        print("Starting training for epoch {}".format(epoch + 1))
        torch.cuda.empty_cache()
        loss_per_epoch = []
        model.train()
        for x, notes_x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            if notes_x is not None:
                notes_x = notes_x.cuda()
            optimizer.zero_grad()
            y_hat = model(x, notes_x)
            y_hat = y_hat.view(y_hat.shape[0]).double()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            loss_per_epoch.append(train_loss)
            # print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
        epoch_loss = statistics.mean(loss_per_epoch)
        print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch + 1, epoch_loss))
        mse = eval_model(model, val_loader)
        mses[epoch] = mse
        train_losses[epoch] = epoch_loss
        gc.collect()
    return train_losses, mses


from torch.utils.data import Dataset


class EpisodeDataset(Dataset):

    def __init__(self, obs, los, index_df, folder="train", mode="both"):
        self.x = obs
        self.y = los
        self.index = index_df
        self.folder = folder
        self.mode = mode

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        idx = self.index[self.index.idx == index]
        # print("Index is ", idx)
        # print("patient_id ", idx.iat[0,1])
        # print("episode ", idx.iat[0,2])
        notes = None
        if self.mode != "physio":
            pe_notes = load_notes_embeddings(idx.iat[0, 1], idx.iat[0, 2], self.folder)
            # print(pe_notes.shape)
            notes = pe_notes[idx.iat[0, 3]]
        return (self.x[index], notes, self.y[index])


if __name__ == "__main__":
    print("main")
    start_time = time.time()
    DATA_PATH = "/mnt/data01/pkj3/length-of-stay/"
    sample = False
    if args.data == "sample":
        sample = True
    X_train, Y_train, index_train = preprocess(
        "train",
        use_saved=args.use_saved,
        window_len=args.window,
        sample=sample,
        sample_size=args.train_sample_size,
    )
    # X_train, Y_train = X_train.to(device), Y_train.to(device)
    train_dataset = EpisodeDataset(
        X_train, Y_train, index_df=index_train, mode=args.mode
    )
    X_val, Y_val, index_val = preprocess(
        "val",
        use_saved=args.use_saved,
        window_len=args.window,
        sample=sample,
        sample_size=args.val_sample_size,
    )
    # X_val, Y_val = X_val.to(device), Y_val.to(device)
    val_dataset = EpisodeDataset(X_val, Y_val, index_df=index_val, mode=args.mode)
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    # model = PhysioNet()
    model = EpisodeNet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    train_losses, val_mses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        optimizer=optimizer,
        criterion=criterion,
    )

    model_path = "{}{}_{}_{}_{}.pt".format(
        args.model_output_dir, args.window, args.lr, args.batch_size, args.epochs
    )
    torch.save(model.state_dict(), model_path)

    # save train losses as png
    matplotlib.use("Agg")
    plt.figure(0)
    plt.xticks([i for i in range(args.epochs)])
    train_losses = [log(y) for y in train_losses]
    plt.plot(train_losses)
    plt.savefig(
        "train_{}_{}_{}_{}.png".format(
            args.window, args.lr, args.batch_size, args.epochs
        )
    )

    plt.figure(1)
    plt.xticks([i for i in range(args.epochs)])
    plt.plot(val_mses)
    plt.savefig(
        "val_{}_{}_{}_{}.png".format(args.window, args.lr, args.batch_size, args.epochs)
    )

    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken : {} secs".format(total_time))
