# A Neural Way to Predict Length of Stay

Group project for CS 598 Deep learning for Healthcare : University of Illinois Urbana-Champaign under Prfessor Jimeng Sun

# Environment Setup

## GPU / CUDA setup

Follow instructions [here](https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1)

Note that CUDA 11.1 was used which is currently compatible with the stable version of pytorch.

Use `watch -n 2 nvidia-smi` to see if the GPU is being used.

## Conda environment (project)

Conda was used to create the development enviornment first creating a new conda environment

`conda create dl4h`

After activating the new environment install pytorch using conda install:

`conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge`

## Conda environment (benchmark)

The benchmark used for comparison requires an environment created with Keras/Tensorflow. The environment must include:

- Python 3.6.8
- Tensorflow
- Keras

Ensure that you create the environment according to requirements.txt

Note that tensorflow is not listed in the requirements. For CPU install the correct version with conda:
`conda install tensorflow`

To run with GPU you will likely need to uninstall tensorflow and then re-install
`pip uninstall tensorflow`

This will also uninstall keras. Reinstall with:
`conda install tensorflow-gpu`

Also re-install keras:
`conda install keras`

To change back to CPU you can uninstall tensorflow using conda and then re-install:
`conda install tensorflow`

# Running the data pre-processing from mimic source

This section show the commands used to run the pre-processing of the data

`python -m mimic3benchmark.scripts.extract_subjects /mnt/data01/mimic-3/csv /mnt/data01/mimic-3/benchmark-notes`

`python -m mimic3benchmark.scripts.validate_events /mnt/data01/mimic-3/benchmark-notes`

`python -m mimic3benchmark.scripts.extract_episodes_from_subjects /mnt/data01/mimic-3/benchmark-notes --notes --notes_csv_file /mnt/data01/mimic-3/csv/NOTEEVENTS.csv`

`python -m mimic3benchmark.scripts.split_train_and_test /mnt/data01/mimic-3/benchmark-notes`

`python -m mimic3benchmark.scripts.create_length_of_stay /mnt/data01/mimic-3/benchmark-notes /mnt/data01/mimic-3/benchmark-notes/length-of-stay/
python -m mimic3models.split_train_val /mnt/data01/mimic-3/benchmark-notes/length-of-stay`

# Running the training

- When running training or any pre-processing activity that will run for a long time and could be interupted. Be sure to send stdout/err to a log file and run in the background. If your terminal dies then your job may end.
- All commands are run from the root project directory
- You can load the state from a checkpoint to continue training from there. The base directory for models is: `/mnt/data01/models`. Use `--load_state` parameter to load a specific state
- You can

## LSTM (From Benchmark)

### partition = custom (classification)

`python -um mimic3models.length_of_stay.main --network mimic3models/keras_models/lstm.py --dim 64 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 256 --partition custom --verbose 1 --save_every 200 --train_batches 500 --val_batches 50 --workers 3 --epochs 18 --data /mnt/data01/mimic-3/benchmark/length-of-stay --output_dir /mnt/data01/models/lstm/custom &> logs/lstm.bins.d64.bs256.log &`

### partition = none (regression)

Same as classification but `--partition` is set to `none`

`python -um mimic3models.length_of_stay.main --network mimic3models/keras_models/lstm.py --dim 64 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 256 --partition none --verbose 1 --save_every 200 --train_batches 500 --val_batches 50 --workers 3 --epochs 18 --data /mnt/data01/mimic-3/benchmark/length-of-stay --output_dir /mnt/data01/models/lstm/regression &> logs/lstm.regression.d64.bs256.log &`

# Running the testing

## LSTM (From Benchmark)

### partition = custom (classification)

`python -um mimic3models.length_of_stay.main --network mimic3models/keras_models/lstm.py --dim 64 --timestep 1.0 --mode test --batch_size 256 --partition custom --verbose 1 --val_batches --workers 3 --data /mnt/data01/mimic-3/benchmark/length-of-stay --output_dir /mnt/data01/models/lstm/custom &> logs/lstm.bins.test.d64.bs256.log &`
