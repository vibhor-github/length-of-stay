python -m mimic3benchmark.scripts.extract_subjects --test /mnt/data01/mimic-3/csv /mnt/data01/mimic-3/benchmark-small
python -m mimic3benchmark.scripts.validate_events /mnt/data01/mimic-3/benchmark-small
python -m mimic3benchmark.scripts.extract_episodes_from_subjects /mnt/data01/mimic-3/benchmark-small --notes --notes_csv_file /mnt/data01/mimic-3/csv/NOTEEVENTS.csv
python -m mimic3benchmark.scripts.split_train_and_test /mnt/data01/mimic-3/benchmark-small
python -m mimic3benchmark.scripts.create_length_of_stay /mnt/data01/mimic-3/benchmark-small /mnt/data01/mimic-3/benchmark-small/length-of-stay/
python -m mimic3models.split_train_val /mnt/data01/mimic-3/benchmark-small/length-of-stay/

