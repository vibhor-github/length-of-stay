import pandas as pd
import numpy as np
import pyarrow as pa
import sys
import spacy
import re
import time
import scispacy
import glob
import os
from tqdm import tqdm
tqdm.pandas()
from note_processing.heuristic_tokenize import sent_tokenize_rules 
from mimic3models.preprocessing import Discretizer_Notes


LOS_PATH = "/mnt/data01/mimic-3/benchmark-notes/length-of-stay"
LISTFILES = ["test_listfile.csv", "train_listfile.csv", "val_listfile.csv"]
# LISTFILES = ["val_listfile.csv"]
NOTEABR = "bert"
EMBEDDIM = 768
WORKERS = 2
TIMESTEP = 5

discretizer = Discretizer_Notes(timestep=TIMESTEP,
                          store_masks=False,  # Not currently supported
                          impute_strategy='previous',
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


def process_episode(tup):
        ts_filename = tup["stay"]
        test_train = tup["set"]

        patient_id = re.findall(r'[0-9]+_', ts_filename)[0][:-1]
        episode = re.findall(r'episode[0-9]+_', ts_filename)[-1][7:-1]
        
        df, filename = get_episode_embeddings(tup)
        
        try:        
            if df is not None:
                df_np = df.to_numpy()

                # Create tensor with impution
                (tensor, header) = discretizer.transform(df_np, header=None, end=int(tup["period_length"]))

                outfile = f"episode{episode}_notes_{NOTEABR}_bin{TIMESTEP}_tensor.parquet"
                out_df = pd.DataFrame([{"TEXT_BIN_EMBEDDING": tensor.tolist()}])
                out_df.to_parquet(os.path.join(os.path.dirname(filename), outfile))
            else:
                outfile = None
        except BaseException as e:
            print(f"Failed to process {patient_id}:{episode} because: {str(e)}")
            outfile = None
        return outfile


## Main
for listfile in LISTFILES:
    filename = os.path.join(LOS_PATH, listfile)
    df = pd.read_csv(filename)
    df["set"] = re.findall(r'(?:test|train|val)', listfile)[0]
    df["set"] = df["set"].apply(lambda x: "train" if x == "val" else x)

    group_df = df.groupby(["stay","set"], as_index=False)["period_length"].agg("max")

    # For each group build the imputed note tensor
    tensor_df = group_df.copy().reset_index()
    tensor_df["tensor"] = tensor_df.progress_apply(process_episode, axis=1)



