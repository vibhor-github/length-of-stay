from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

from mimic3benchmark.subject import read_stays, read_diagnoses, read_events, get_events_for_stay,\
    add_hours_elpased_to_events
from mimic3benchmark.subject import convert_events_to_timeseries, get_first_valid_from_timeseries
from mimic3benchmark.preprocessing import read_itemid_to_variable_map, map_itemids_to_variables, clean_events
from mimic3benchmark.preprocessing import assemble_episodic_data

def get_icu_stay_from_dt_func(stays):
    def get_icu_stay(chartdate):
        for i in range(len(stays["INTIME"])):
            latest_start = max(stays["INTIME"].iloc[i], chartdate)
            earliest_end = min(stays["OUTTIME"].iloc[i], chartdate)
            if earliest_end < latest_start:
                continue
            # delta = (earliest_end - latest_start)
            # delta = delta.days*24 + delta.seconds/3600
            # overlap = max(0, delta)
            # if overlap > 0:
            else:
                return int(stays["ICUSTAY_ID"].iloc[i])
        return np.nan
    return get_icu_stay

parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
parser.add_argument('--variable_map_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/itemid_to_variable_map.csv'),
                    help='CSV containing ITEMID-to-VARIABLE map.')
parser.add_argument('--reference_range_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/variable_ranges.csv'),
                    help='CSV containing reference ranges for VARIABLEs.')
parser.add_argument('--notes', action='store_true', help='NOTES: Process notes')
parser.add_argument('--notes_csv_file', type=str,
                    help='CSV file with all mimic clinical notes')
args, _ = parser.parse_known_args()

var_map = read_itemid_to_variable_map(args.variable_map_file)
variables = var_map.VARIABLE.unique()

if args.notes:
    all_notes = pd.read_csv(args.notes_csv_file, 
                            parse_dates=["CHARTTIME"], 
                            infer_datetime_format=True)
    #TODO If Charttime is missing set from the chartdate
    #notes.CHARTTIME.fillna(notes.CHARTDATE,inplace=True)
    all_notes.drop(all_notes[all_notes.ISERROR == 1].index, inplace=True)
    all_notes.drop(columns="ISERROR", inplace=True)

for subject_dir in tqdm(os.listdir(args.subjects_root_path), desc='Iterating over subjects'):
    dn = os.path.join(args.subjects_root_path, subject_dir)
    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except:
        continue

    try:
        # reading tables of this subject
        stays = read_stays(os.path.join(args.subjects_root_path, subject_dir))
        diagnoses = read_diagnoses(os.path.join(args.subjects_root_path, subject_dir))
        events = read_events(os.path.join(args.subjects_root_path, subject_dir))
    except:
        sys.stderr.write('Error reading from disk for subject: {}\n'.format(subject_id))
        continue

    episodic_data = assemble_episodic_data(stays, diagnoses)

    # cleaning and converting to time series
    events = map_itemids_to_variables(events, var_map)
    events = clean_events(events)
    if events.shape[0] == 0:
        # no valid events for this subject
        continue
    timeseries = convert_events_to_timeseries(events, variables=variables)

    if args.notes:
        notes = all_notes[(all_notes["SUBJECT_ID"] == subject_id) & (~pd.isnull(all_notes["CHARTTIME"]))] \
                    .sort_values(["CHARTTIME"])
        notes["ICUSTAY_ID"] = notes['CHARTTIME'].apply(get_icu_stay_from_dt_func(stays))
        notes = notes[notes['ICUSTAY_ID'].notna()]

    # extracting separate episodes
    for i in range(stays.shape[0]):
        stay_id = stays.ICUSTAY_ID.iloc[i]
        intime = stays.INTIME.iloc[i]
        outtime = stays.OUTTIME.iloc[i]

        episode = get_events_for_stay(timeseries, stay_id, intime, outtime)
        if episode.shape[0] == 0:
            # no data for this episode
            continue

        episode = add_hours_elpased_to_events(episode, intime).set_index('HOURS').sort_index(axis=0)
        if stay_id in episodic_data.index:
            episodic_data.loc[stay_id, 'Weight'] = get_first_valid_from_timeseries(episode, 'Weight')
            episodic_data.loc[stay_id, 'Height'] = get_first_valid_from_timeseries(episode, 'Height')
        episodic_data.loc[episodic_data.index == stay_id].to_csv(os.path.join(args.subjects_root_path, subject_dir,
                                                                              'episode{}.csv'.format(i+1)),
                                                                 index_label='Icustay')
        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
        episode = episode[columns_sorted]
        episode.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}_timeseries.csv'.format(i+1)),
                       index_label='Hours')

        # aflanders: save notes
        if args.notes:
            event_notes = notes[(notes["ICUSTAY_ID"] == stay_id)].copy()
            event_notes['HOURS'] = (event_notes.CHARTTIME - intime).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60
            event_notes = event_notes[["HOURS", "CATEGORY", "DESCRIPTION", "TEXT"]].set_index('HOURS').sort_index(axis=0)
            event_notes.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}_notes.csv'.format(i+1)),
                            index_label='Hours')