# aflanders:
# Processes note files generated durring pre-processing and converts them
# into sentences which can be used for embeddings and downstream training
# This code will split the notes into natural sentence boundaries separated by \n
# which can then be fed into sentence embedding models such as BIO-ClinicalBert or 
# BioSentVec
#
# Some of the code is taken from format_mimic_for_BERT.py in EmilyAlsentzer/clinicalBERT
# I have updated the code to work with spacy 3.0 and made some other changes
#
# Will read all /<patient>/episode<#>_notes.csv files and create a new 
# /<patient>/episode<#>_notes_sent.csv file with sentences broken by \n in TEXT
# 
# Example:
# THis is a 
# single 
# sentence. and another sentence.
#
# THis is a single sentence.\n
# and another sentence.\n

import pandas as pd
import sys
import spacy
from spacy.language import Language
import re
import time
import scispacy
import glob
import os
from tqdm import tqdm
from note_processing.heuristic_tokenize import sent_tokenize_rules 

from pandarallel import pandarallel


MIMIC_NOTES_PATHS = ['/mnt/data01/mimic-3/benchmark-small/test',
                     '/mnt/data01/mimic-3/benchmark-small/train']  
WORKERS = 10

pandarallel.initialize(progress_bar=True, nb_workers=WORKERS)


#category = ["Nursing", "Nursing/other", 'General', 'Physician ']  # or None
#category = ["Nursing/other"]  # or None
category = None

#setting sentence boundaries
@Language.component('sbd_component')
def sbd_component(doc):
    for i, token in enumerate(doc[:-2]):
        # define sentence start if period + titlecase token
        if token.text == '.' and doc[i+1].is_title:
            doc[i+1].sent_start = True
        if token.text == '-' and doc[i+1].text != '-':
            doc[i+1].sent_start = True
    return doc

#convert de-identification text into one token
def fix_deid_tokens(doc):
    deid_regex  = r"\[\*\*.{0,15}.*?\*\*\]" 

    indexes = [m.span() for m in re.finditer(deid_regex, doc.text, flags=re.IGNORECASE)]

    for start,end in indexes:
        # processed_text.merge(start_idx=start,end_idx=end)
        # aflanders: Make compatible with latest version fo spacy
        try:
            span = doc.char_span(start, end)
            if span is not None:
                with doc.retokenize() as retokenizer:
                    # retokenizer.merge(processed_text[start:end+1])
                    retokenizer.merge(span)
        except:
            print(f'Error with: {text}')
                
    return doc

def process_section(section, note, processed_sections):
    # perform spacy processing on section
    processed_section = nlp(section['sections'])
    # processed_section = fix_deid_tokens(section['sections'], processed_section)
    processed_section = fix_deid_tokens(processed_section)
    processed_sections.append(processed_section)

def process_note_helper(note):
    # split note into sections
    note_sections = sent_tokenize_rules(note)
    processed_sections = []
    section_frame = pd.DataFrame({'sections':note_sections})
    section_frame.apply(process_section, args=(note,processed_sections,), axis=1)
    return(processed_sections)

def process_text(sent, note):
    sent_text = sent['sents'].text
    if len(sent_text) > 0 and sent_text.strip() != '\n':
        if '\n' in sent_text:
            sent_text = sent_text.replace('\n', ' ')
        note['TEXT'] += sent_text + '\n'  

def get_sentences(processed_section, note):
    # get sentences from spacy processing
    sent_frame = pd.DataFrame({'sents': list(processed_section['sections'].sents)})
    sent_frame.apply(process_text, args=(note,), axis=1)

def process_note(note):
    # try:
        note_text = note['TEXT'] #unicode(note['TEXT'])
        note['TEXT'] = ''
        processed_sections = process_note_helper(note_text)
        ps = {'sections': processed_sections}
        ps = pd.DataFrame(ps)
        ps.apply(get_sentences, args=(note,), axis=1)
        return note 
    # except Exception as e:
    #     # pass
    #     print ('error processing note', e)


####### MAIN ####### 
all_files = []

for path in MIMIC_NOTES_PATHS:
    files = glob.glob(path + "/*/*_notes.csv")
    all_files += files

print("\nTotal note files: " + str(len(all_files)))
all_files = [f for f in all_files if not os.path.exists(f[:-4] + '_sent.csv')]
print("Total unprocessed files: " + str(len(all_files)))

li = []

for filename in tqdm(all_files, desc="Load note files"):
    df = pd.read_csv(filename, index_col=None, header=0)
    df["filename"] = filename
    li.append(df)

notes = pd.concat(li, axis=0, ignore_index=True)

# Filter categories
if category != None:
    notes = notes[notes['CATEGORY'].isin(category)]
print('Number of notes: %d' %len(notes.index))
if len(notes.index) < 1:
    print('No notes to process')
    quit()

# notes['ind'] = list(range(len(notes.index)))

nlp = spacy.load('en_core_sci_md', disable=['tagger','ner', 'lemmatizer'])
nlp.add_pipe('sbd_component', before='parser')  

# Process the notes
print('Begin reading notes')
formatted_notes = notes.parallel_apply(process_note, axis=1)

# Write out a new note files organized by sentence
print("This is the type: " + str(type(formatted_notes)))
filenames = list(formatted_notes["filename"].unique().tolist())
for filename in tqdm(filenames, desc="Writing note sentence files"):
    df = formatted_notes[formatted_notes["filename"] == filename][["Hours", "CATEGORY", "DESCRIPTION", "TEXT"]]
    df = df.set_index("Hours")
    write_file = filename.replace(".csv", "_sent.csv")
    with open(write_file, "w") as f:
        df.to_csv(f, index_label='Hours')