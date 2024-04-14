import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import pandas as pd
from preprocess import preprocess
from model import EpisodeNet
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from tqdm import trange
import time
from torch.utils.data import Dataset
from main import EpisodeDataset
from main import eval_model


import argparse

test_parser = argparse.ArgumentParser()
test_parser.add_argument('--batch_size', type=int, default = 128, help= "batch size")
test_parser.add_argument('--window', type=int, default=5, help="minimum 5")
test_parser.add_argument('--use_saved', type=bool, default=True, help="use saved embeddings")
test_parser.add_argument('--data', type=str, help='sample / all',default="all")

test_parser.add_argument('--test_sample_size', type=int, help='if data == sample, provide training sample size',default=5000)
test_parser.add_argument('--model_output_dir', type=str, help='Model output dir',default='/mnt/data01/models/cnn/')
test_parser.add_argument('--model_name', type=str, help='Model output dir')
test_args = test_parser.parse_args()
print(test_args)


if __name__ == "__main__":
    print("Testing")
    start_time = time.time()
    MODEL_PATH =  test_args.model_output_dir+test_args.model_name
    print("Loading model {}".format(MODEL_PATH))
    if test_args.data == 'sample':
        sample = True
    X_test, Y_test, test_index = preprocess('test', use_saved = test_args.use_saved, window_len = test_args.window, sample = sample, sample_size = test_args.test_sample_size)
    test_dataset = EpisodeDataset(X_test, Y_test, test_index, folder = "test")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EpisodeNet()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_args.batch_size,shuffle=False)                            
    eval_model(model, test_loader)
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken : {} secs".format(total_time))