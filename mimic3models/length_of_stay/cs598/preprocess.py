import json
import numpy as np
import torch
import os
import pandas as pd
from tqdm import tqdm
from functools import lru_cache

replacement_map = None
default_value_map = None
prev_value_map = {}
DATA_PATH = "/mnt/data01/pkj3/length-of-stay/"
SAVED_TENSOR_PATH = DATA_PATH + "saved/"
TRAIN_PATH = DATA_PATH + "train/"
TEST_PATH = DATA_PATH + "test/"
train_y_df = pd.read_csv(DATA_PATH + "train_listfile.csv")
train_files = train_y_df["stay"].unique().tolist()
val_y_df = pd.read_csv(DATA_PATH + "val_listfile.csv")
val_files = val_y_df["stay"].unique().tolist()
test_y_df = pd.read_csv(DATA_PATH + "test_listfile.csv")
test_files = test_y_df["stay"].unique().tolist()
train_y_min, train_y_max = train_y_df.y_true.describe(percentiles=[0.03, 0.97])[
    ["3%", "97%"]
].to_list()
val_y_min, val_y_max = val_y_df.y_true.describe(percentiles=[0.03, 0.97])[
    ["3%", "97%"]
].to_list()
test_y_min, test_y_max = test_y_df.y_true.describe(percentiles=[0.03, 0.97])[
    ["3%", "97%"]
].to_list()
train_y_df = train_y_df[
    (train_y_df.y_true > train_y_min) & (train_y_df.y_true < train_y_max)
]
val_y_df = val_y_df[(val_y_df.y_true > val_y_min) & (val_y_df.y_true < val_y_max)]
test_y_df = test_y_df[(test_y_df.y_true > test_y_min) & (test_y_df.y_true < test_y_max)]

with open("config.json") as f:
    config = json.load(f)
    replacement_map = config["replacement_map"]
    default_value_map = config["default_value_map"]


def to_np(elem):
    return np.concatenate([np.array(i) for i in elem])


def cleanup(episode_df):
    episode_df["Glascow coma scale eye opening"] = episode_df[
        "Glascow coma scale eye opening"
    ].apply(
        lambda x: (
            replacement_map["Glascow coma scale eye opening"][x]
            if x in replacement_map["Glascow coma scale eye opening"]
            else x
        )
    )
    episode_df["Glascow coma scale motor response"] = episode_df[
        "Glascow coma scale motor response"
    ].apply(
        lambda x: (
            replacement_map["Glascow coma scale motor response"][x]
            if x in replacement_map["Glascow coma scale motor response"]
            else x
        )
    )
    episode_df["Glascow coma scale verbal response"] = episode_df[
        "Glascow coma scale verbal response"
    ].apply(
        lambda x: (
            replacement_map["Glascow coma scale verbal response"][x]
            if x in replacement_map["Glascow coma scale verbal response"]
            else x
        )
    )


def process_column(person_id, value, colname):
    if value is not None and not np.isnan(value):
        prev_value_map[person_id][colname] = value
        return value
    if (
        person_id in prev_value_map
        and colname in prev_value_map[person_id]
        and prev_value_map[person_id][colname] is not None
    ):
        prev = prev_value_map[person_id][colname]
    else:
        prev = default_value_map[colname]
    return prev


def fill_missing_values(pateint_id, episode_df):
    prev_value_map[pateint_id] = {}
    episode_df["Capillary refill rate"] = episode_df["Capillary refill rate"].apply(
        lambda x: process_column(pateint_id, x, "Capillary refill rate")
    )
    episode_df["Diastolic blood pressure"] = episode_df[
        "Diastolic blood pressure"
    ].apply(lambda x: process_column(pateint_id, x, "Diastolic blood pressure"))
    episode_df["Fraction inspired oxygen"] = episode_df[
        "Fraction inspired oxygen"
    ].apply(lambda x: process_column(pateint_id, x, "Fraction inspired oxygen"))
    episode_df["Glascow coma scale eye opening"] = episode_df[
        "Glascow coma scale eye opening"
    ].apply(lambda x: process_column(pateint_id, x, "Glascow coma scale eye opening"))
    episode_df["Glascow coma scale motor response"] = episode_df[
        "Glascow coma scale motor response"
    ].apply(
        lambda x: process_column(pateint_id, x, "Glascow coma scale motor response")
    )
    episode_df["Glascow coma scale total"] = episode_df[
        "Glascow coma scale total"
    ].apply(lambda x: process_column(pateint_id, x, "Glascow coma scale total"))
    episode_df["Glascow coma scale verbal response"] = episode_df[
        "Glascow coma scale verbal response"
    ].apply(
        lambda x: process_column(pateint_id, x, "Glascow coma scale verbal response")
    )
    episode_df["Glucose"] = episode_df["Glucose"].apply(
        lambda x: process_column(pateint_id, x, "Glucose")
    )
    episode_df["Heart Rate"] = episode_df["Heart Rate"].apply(
        lambda x: process_column(pateint_id, x, "Heart Rate")
    )
    episode_df["Mean blood pressure"] = episode_df["Mean blood pressure"].apply(
        lambda x: process_column(pateint_id, x, "Mean blood pressure")
    )
    episode_df["Height"] = episode_df["Height"].apply(
        lambda x: process_column(pateint_id, x, "Height")
    )
    episode_df["Oxygen saturation"] = episode_df["Oxygen saturation"].apply(
        lambda x: process_column(pateint_id, x, "Oxygen saturation")
    )
    episode_df["Respiratory rate"] = episode_df["Respiratory rate"].apply(
        lambda x: process_column(pateint_id, x, "Respiratory rate")
    )
    episode_df["Systolic blood pressure"] = episode_df["Systolic blood pressure"].apply(
        lambda x: process_column(pateint_id, x, "Systolic blood pressure")
    )
    episode_df["Temperature"] = episode_df["Temperature"].apply(
        lambda x: process_column(pateint_id, x, "Temperature")
    )
    episode_df["Weight"] = episode_df["Weight"].apply(
        lambda x: process_column(pateint_id, x, "Weight")
    )
    episode_df["pH"] = episode_df["pH"].apply(
        lambda x: process_column(pateint_id, x, "pH")
    )
    del prev_value_map[pateint_id]


def get_window_indices(data_len, window_len=4):
    i = 0
    indices = []
    while i <= data_len - window_len:
        indices.append([i + j for j in range(window_len)])
        i += 1
    return indices


def get_one_hot_encoding(val, dim):
    encoding = [0 for i in range(dim)]
    encoding[val - 1] = 1
    return encoding


def preprocess(path, use_saved=True, window_len=5, sample=False, sample_size=5000):
    x_file_name = "{}_{}_{}_X.pt".format(path, window_len, sample_size)
    y_file_name = "{}_{}_{}_Y.pt".format(path, window_len, sample_size)
    idx_file_name = "{}_{}_{}_idx.csv".format(path, window_len, sample_size)
    if use_saved:
        print("Attempting to use saved files")
        saved_files = os.listdir(SAVED_TENSOR_PATH)
        print("Saved files : ", saved_files)
        if x_file_name in saved_files and y_file_name in saved_files:
            print(
                "Loading X and Y from saved files: {} {}. Will Skip processing".format(
                    x_file_name, y_file_name
                )
            )
            X = torch.load(SAVED_TENSOR_PATH + x_file_name)
            Y = torch.load(SAVED_TENSOR_PATH + y_file_name)
            index_df = pd.read_csv(SAVED_TENSOR_PATH + idx_file_name)
            return (X, Y, index_df)
    print("Processing " + path + " files.")
    preprocess_path = TRAIN_PATH
    if path == "train":
        y_df = train_y_df
        if sample:
            data_files = np.random.choice(train_files, sample_size, replace=False)
        else:
            data_files = train_files
    elif path == "val":
        y_df = val_y_df
        if sample:
            data_files = np.random.choice(val_files, sample_size, replace=False)
        else:
            data_files = val_files
    else:
        y_df = test_y_df
        preprocess_path = TEST_PATH
        if sample:
            data_files = np.random.choice(test_files, sample_size, replace=False)
        else:
            data_files = test_files

    # x_path = DATA_PATH +'/'+path+'/'
    X = torch.empty(0, 34, window_len)
    Y = torch.empty(
        0,
    )
    # y_df = pd.read_csv(DATA_PATH + path +'_listfile.csv')
    # data_files = os.listdir(x_path)
    print("Number of " + path + " files = ", len(data_files))
    print("Reading from path : {}".format(preprocess_path))
    i = 0
    index_data = []
    for data_file in tqdm(data_files):
        if data_file.endswith(".csv"):
            data_info = data_file.split("_")
            episode_df = pd.read_csv(preprocess_path + data_file)
            cleanup(episode_df)
            fill_missing_values(data_file, episode_df)
            episode_df["H_IDX"] = episode_df.Hours.apply(np.floor).astype("int32")
            episode_df = episode_df.groupby(by="H_IDX").agg("last")
            gcser = pd.DataFrame(
                episode_df["Glascow coma scale eye opening"]
                .astype("int32")
                .apply(get_one_hot_encoding, args=(5,))
                .to_list()
            )
            gcsmr = pd.DataFrame(
                episode_df["Glascow coma scale motor response"]
                .astype("int32")
                .apply(get_one_hot_encoding, args=(7,))
                .to_list()
            )
            gcsvr = pd.DataFrame(
                episode_df["Glascow coma scale verbal response"]
                .astype("int32")
                .apply(get_one_hot_encoding, args=(7,))
                .to_list()
            )
            episode_df = pd.concat((episode_df, gcser, gcsmr, gcsvr), axis=1)
            episode_df = episode_df.drop(
                [
                    "Glascow coma scale eye opening",
                    "Glascow coma scale motor response",
                    "Glascow coma scale verbal response",
                ],
                axis=1,
            )
            # episode_df = episode_df[episode_df.Hours>=5].reset_index(drop = True)
            temp_y = (
                y_df[y_df.stay == data_file]
                .sort_values(by="period_length")
                .reset_index(drop=True)
            )
            temp_y = temp_y[["period_length", "y_true"]].set_index("period_length")
            episode_df = episode_df.join(temp_y, how="inner").reset_index(drop=True)
            episode_df = episode_df.dropna().reset_index(drop=True)
            if len(episode_df) > 0:
                indices = get_window_indices(len(episode_df), window_len)
                windows = []
                y_values = []
                for j, idx in enumerate(indices):
                    window = episode_df.loc[idx]
                    y_values.append(window.loc[idx[-1]].y_true)
                    windows.append(
                        window.drop("y_true", axis=1)
                        .transpose()
                        .values.astype(np.float32)
                    )
                    index_data.append((i, data_info[0], data_info[1], j))
                    i = i + 1
                t_windows = torch.tensor(windows)
                t_y_values = torch.tensor(y_values)
                X = torch.cat((X, t_windows), 0)
                Y = torch.cat((Y, t_y_values), 0)
    torch.save(X, SAVED_TENSOR_PATH + x_file_name)
    torch.save(Y, SAVED_TENSOR_PATH + y_file_name)
    index_df = pd.DataFrame(index_data)
    index_df.columns = ["idx", "patient_id", "episode", "window_idx"]
    index_df.to_csv(idx_file_name, index=False)
    return (X, Y, index_df)
