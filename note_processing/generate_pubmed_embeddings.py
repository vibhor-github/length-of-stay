# bhaskarp, rdebroy, aflanders:
# Generate BioSentVec embeddings from clinical notes.
#
# Will read all /<patient>/episode<#>_notes_sent.csv files and create a new 
# /<patient>/episode<#>_notes_BioSentVec.parquet file multi-dimensional BioSentVec embedidngs.
# Shape of the embeddings is (#sentences, 700)
# 

import pandas as pd
import numpy as np
import sys
import spacy
import re
import time
import scispacy
import sent2vec
import glob
import os
from tqdm import tqdm
tqdm.pandas()
from note_processing.heuristic_tokenize import sent_tokenize_rules 
#from heuristic_tokenize import sent_tokenize_rules 

# OUTPUT_DIR = Currently same as the input directory

#'/mnt/data01/mimic-3/benchmark-notes/train'
MIMIC_NOTES_PATHS = ['/mnt/data01/mimic-3/benchmark-notes/test', '/mnt/data01/mimic-3/benchmark-notes/train']  
DEVICE = -1  # -1 is CPU otherwise the GPU device id
MAX_SENT = 40
MAX_TOKEN_LEN = 40
BATCH_SIZE = 200
MODEL_PATH = "/home/bhaskarp/data01/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"


category = ["Nursing", "Nursing/other", "Radiology"]  # or None
# category = ["Nursing/other"]  # or None
#category = None


def shorten_text(sent, _max=100):
    max = _max
    tokens = sent.split()
    if len(tokens) < max:
        ret = sent
    else:
        tokens = tokens[:max]
        ret =  " ".join(tokens)

    return ret


def get_embeddings(text, model, max_sents=200, max_len=100):
    text = str(text)
    sents = text.split('\n')[:-1]
    before_len = len(sents)
    sents = [x for x in sents if len(x.split()) > 1]
    sents = list(map(lambda x: shorten_text(x, max_len), sents))
    sents = sents[:max_sents]
    if len(sents) == 0:
        return None
    #TODO Remove hack
    if len(sents) == 1:
        sents.append("the book")

    while True:
        try:
            sent_features = model.embed_sentences(sents)
        except BaseException as e:
            sent_len = [len(x) for x in sents]
            sent_features = None
            break
        break

    return sent_features


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def generate_embedding(notes, model):
    notes["TEXT_EMBEDDING"] = notes["TEXT"].progress_apply(get_embeddings, args=(model,MAX_SENT,MAX_TOKEN_LEN))

    # Convert to mult-dimensional list for export to parquet
    notes["TEXT_EMBEDDING"] = notes["TEXT_EMBEDDING"].apply(lambda x: x.tolist() if x is not None else [[]])

    # Write out a new notes file with the embeddings
    filenames = list(notes["filename"].unique().tolist())
    for filename in tqdm(filenames, desc="Writing embedding files"):
        df = notes[notes["filename"] == filename][["Hours", "CATEGORY", "DESCRIPTION", "TEXT_EMBEDDING"]]
        df = df.set_index("Hours")
        write_file = filename.replace("_notes_sent.csv", "_notes_BioSentVec.parquet")
        df.to_parquet(write_file)


### MAIN ###
start_time = time.time()

# Load the BioSentVec model
print('Load the BioSentVec model')
model = sent2vec.Sent2vecModel()
try:
    model.load_model(MODEL_PATH)
except Exception as e:
    print(e)
print('model successfully loaded')

# Read all files in the input directories that match the input file format
all_files = []
for path in MIMIC_NOTES_PATHS:
    files = glob.glob(path + "/*/*_notes_sent.csv")
    all_files += files

print(f"\nTotal note files: {len(all_files)}")

for i, filenames in enumerate(batch(all_files, n=BATCH_SIZE)):
    li = []
    for filename in tqdm(filenames, desc=f"Load note file batch: {i}"):
        df = pd.read_csv(filename, index_col=None, header=0)
        df["filename"] = filename
        li.append(df)

    notes = pd.concat(li, axis=0, ignore_index=True)

    print(f'Number of notes: {len(notes.index)}')

    # Restrict the number of notes for processing
    if category != None:
        notes = notes[notes['CATEGORY'].isin(category)].copy()

    print(f'Number of notes for categories {category}: {len(notes.index)}')

    generate_embedding(notes, model)


print(f"Finished in {int(time.time() - start_time)} seconds")