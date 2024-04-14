# aflanders: Script to create pre-processed tensors with windowing scheme

import pandas as pd
import numpy as np
import pyarrow as pa
import sys
import re
import time
import glob
import os
from tqdm import tqdm
tqdm.pandas()

#from note_processing.heuristic_tokenize import sent_tokenize_rules 
from mimic3models.preprocessing import Discretizer_Notes



LOS_PATH = "/mnt/data01/mimic-3/benchmark-notes/length-of-stay"
LISTFILES = ["test_listfile.csv", "train_listfile.csv", "val_listfile.csv"]
#LISTFILES = ["train_listfile.csv"]
NOTEABR = "bert"    # bert or BioSent2Vec
MAX_SENT = 80       # Max sentence len for window
EMBEDDIM = 768      # Embedding dimesions
TIMESTEP = 1        # Timestep for discretizer (not for window)
WINDOW = 5          # Window size
STEPSIZE = 1        # Step size for window

discretizer = Discretizer_Notes(timestep=TIMESTEP,
                          store_masks=False,  # Not currently supported
                          impute_strategy='zero',
                          start_time='zero',
                          sent_dim=80)


def get_episode_embeddings(tup):
        ts_filename = tup["stay"]
        test_train = tup["set"]

        ret = []
        patient_id = re.findall(r'[0-9]+_', ts_filename)[0][:-1]
        episode = re.findall(r'episode[0-9]+_', ts_filename)[-1][7:-1]

        par_dir = os.path.abspath(os.path.join(LOS_PATH, os.pardir))

        filename = f"episode{episode}_notes_{NOTEABR}.parquet"
        filename = os.path.join(par_dir, test_train, patient_id, filename)
        
        columns = ["Hours", "CATEGORY", "DESCRIPTION", "TEXT_EMBEDDING"]
        try:
            df = pd.read_parquet(filename)
            columns = list(df.columns)
            df["Hours"] = df.index
            columns.insert(0, "Hours")
            ret = df[columns]
        except BaseException as e:
            print(f"Fail for patient: {patient_id} with error: {str(e)}")
            # TODO Remove hack
            ret = None

        return ret, filename


def pack_window(tensors):
    data = np.zeros((MAX_SENT, EMBEDDIM))
    tensors = np.flip(tensors, 0)

    mask = [0] * WINDOW
    for i, row in enumerate(tensors):
        embedding = row

        if row[0,0] == 0:
            continue

        start = 0
        while start < MAX_SENT and data[start, 0] != 0:
            start += 1

        if start >= MAX_SENT:
            # DROPPED_SENTS += embedding.shape[0]
            continue

        over = (start + embedding.shape[0]) - MAX_SENT
        # DROPPED_SENTS += over if over > 0 else 0
        remaining = embedding.shape[0] if over < 0 else MAX_SENT - start
        remain_embedding = np.stack(embedding[:remaining])
        data[start:start+remaining] = remain_embedding
        mask[i] = 1

    return data, mask


def create_windows(tensor):
    windows = np.zeros((int((tensor.shape[0]-(WINDOW-1))/STEPSIZE), MAX_SENT, EMBEDDIM))
    # print(f"Tensor shape is {tensor.shape}")
    # print(f"Windows shape is {windows.shape}")
    masks = []

    for i in range(0, windows.shape[0], STEPSIZE):
        window, mask = pack_window(tensor[i:i+WINDOW])
        windows[i] = window
        masks.append(mask)

    masks = np.stack(masks)

    return windows, masks


def process_episode(tup):
        ts_filename = tup["stay"]
        test_train = tup["set"]

        patient_id = re.findall(r'[0-9]+_', ts_filename)[0][:-1]
        episode = re.findall(r'episode[0-9]+_', ts_filename)[-1][7:-1]
        par_dir = os.path.abspath(os.path.join(LOS_PATH, os.pardir))

        outfile = f"episode{episode}_notes_{NOTEABR}_window{WINDOW}-{STEPSIZE}_tensor.parquet"
        path = os.path.join(par_dir, test_train, patient_id)
        if os.path.exists(os.path.join(path, outfile)) or not os.path.exists(os.path.join(path, f"episode{episode}_notes_{NOTEABR}.parquet")):
            return outfile
        
        df, filename = get_episode_embeddings(tup)
        
        try:
            if df is not None:
                df_np = df.to_numpy() # (#notes, #sent(var), #embedding)

                # Create tensor with impution
                (tensor, header) = discretizer.transform(df_np, header=None, end=int(tup["period_length"]))
                tensor, masks = create_windows(tensor)

                out_df = pd.DataFrame([{"TEXT_WINDOW_EMBEDDING": tensor.tolist()}])
                out_df.to_parquet(os.path.join(path, outfile))
            else:
                outfile = None
        except BaseException as e:
            print(f"Failed to process {patient_id}:{episode} because: {str(e)}")
            outfile = None

        return outfile
  

## Main ##
for listfile in LISTFILES:
    filename = os.path.join(LOS_PATH, listfile)
    df = pd.read_csv(filename)
    df["set"] = re.findall(r'(?:test|train|val)', listfile)[0]
    df["set"] = df["set"].apply(lambda x: "train" if x == "val" else x)

    group_df = df.groupby(["stay","set"], as_index=False)["period_length"].agg("max")

    # For each group build the imputed note tensor
    tensor_df = group_df.copy().reset_index()
    tensor_df["tensor"] = tensor_df.progress_apply(process_episode, axis=1)

