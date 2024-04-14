from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re

from mimic3models.length_of_stay import utils
from mimic3benchmark.readers import LengthOfStayReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from tqdm import tqdm


parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--deep_supervision', dest='deep_supervision', action='store_true')
parser.set_defaults(deep_supervision=False)
parser.add_argument('--partition', type=str, default='custom',
                    help="log, custom, none")
parser.add_argument('--data', type=str, help='Path to the data of length-of-stay task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/length-of-stay/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')

# aflanders: Add additional arguments to tune training
parser.add_argument('--train_batches', nargs='?', type=int, default=500,
    help='number of batches to train on per epoch. Set param without value to train on all')
parser.add_argument('--val_batches', nargs='?', type=int, default=200,
    help='Number of batches to validate or test. Set parm without value for full test/validation')
parser.add_argument('--workers', type=int, default=1,
    help='number of workers to load data')
parser.add_argument('--stop_early_loss', type=float, default=0.001,
    help='Minimum loss gain')

args = parser.parse_args()
print(args)

if args.small_part:
    args.save_every = 2**30

# Build readers, discretizers, normalizers
if args.deep_supervision:
    train_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'train'),
                                                               listfile=os.path.join(args.data, 'train_listfile.csv'),
                                                               small_part=args.small_part)
    val_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'train'),
                                                             listfile=os.path.join(args.data, 'val_listfile.csv'),
                                                             small_part=args.small_part)
else:
    train_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'),
                                      listfile=os.path.join(args.data, 'train_listfile.csv'))
    val_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'),
                                    listfile=os.path.join(args.data, 'val_listfile.csv'))

discretizer = Discretizer(timestep=args.timestep,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

if args.deep_supervision:
    discretizer_header = discretizer.transform(train_data_loader._data["X"][0])[1].split(',')
else:
    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'los_ts{}.input_str:previous.start_time:zero.n5e4.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'los'
args_dict['num_classes'] = (1 if args.partition == 'none' else 10)


# Build the model
print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
suffix = "{}.bs{}{}{}.ts{}.partition={}".format("" if not args.deep_supervision else ".dsup",
                                                args.batch_size,
                                                ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                                ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                                args.timestep,
                                                args.partition)
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)


# Compile the model
print("==> compiling the model")
# TODO: aflanders: Determine how to pass the learning rate and beta arguments into the optimizer.
# There was a change in tensorflow 2.0
# optimizer_config = {'class_name': args.optimizer,
#                     'config': {'learning_rate': args.lr,
#                                'beta_1': args.beta_1}}

optimizer_config = args.optimizer

if args.partition == 'none':
    # other options are: 'mean_squared_error', 'mean_absolute_percentage_error'
    loss_function = 'mean_squared_logarithmic_error'
else:
    loss_function = 'sparse_categorical_crossentropy'
# NOTE: categorical_crossentropy needs one-hot vectors
#       that's why we use sparse_categorical_crossentropy
# NOTE: it is ok to use keras.losses even for (B, T, D) shapes

model.compile(optimizer=optimizer_config,
            loss=loss_function)
model.summary()


# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*chunk([0-9]+).*", args.load_state).group(1))

if args.mode == 'train':
    # Load data and prepare generators
    if args.deep_supervision:
        train_data_gen = utils.BatchGenDeepSupervision(train_data_loader, args.partition,
                                                    discretizer, normalizer, args.batch_size, shuffle=True)
        val_data_gen = utils.BatchGenDeepSupervision(val_data_loader, args.partition,
                                                    discretizer, normalizer, args.batch_size, shuffle=False)
    else:
        # Set number of batches in one epoch
        # aflanders: Make the step count a parameter. The steps have been changed to be
        # one batch size each
        # train_nbatches = 2000
        # val_nbatches = 1000
        train_nbatches = args.train_batches
        val_nbatches = args.val_batches

        if args.small_part:
            train_nbatches = 20
            val_nbatches = 20

        train_data_gen = utils.BatchGen(reader=train_reader,
                                        discretizer=discretizer,
                                        normalizer=normalizer,
                                        partition=args.partition,
                                        batch_size=args.batch_size,
                                        steps=train_nbatches,
                                        shuffle=True,
                                        verbose=args.verbose)
        val_data_gen = utils.BatchGen(reader=val_reader,
                                    discretizer=discretizer,
                                    normalizer=normalizer,
                                    partition=args.partition,
                                    batch_size=args.batch_size,
                                    steps=val_nbatches,
                                    shuffle=False,
                                    verbose=args.verbose)

    # Prepare training
    #aflanders: Make compatible with Tensorflow 2.0
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.chunk{epoch:02d}.test{loss:.4f}.state')

    metrics_callback = keras_utils.LengthOfStayMetrics(train_data_gen=train_data_gen,
                                                       val_data_gen=val_data_gen,
                                                       partition=args.partition,
                                                       batch_size=args.batch_size,
                                                       verbose=args.verbose,
                                                       workers=args.workers)
    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    #aflanders: Tensorflow 2.0 compatibility
    saver = ModelCheckpoint(path, verbose=1, save_freq=args.save_every, save_weights_only=True)
    earlyStopping = EarlyStopping(monitor='loss', min_delta=args.stop_early_loss, verbose=args.verbose)
    #aflanders: remove warning
    
    keras_logs = os.path.join(args.output_dir, 'keras_logs')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> training")
    #aflanders: Change to use multi-processing and support tensorflow 2.0
    # model.fit_generator(generator=train_data_gen,
    #                     steps_per_epoch=train_data_gen.steps,
    #                     validation_data=val_data_gen,
    #                     validation_steps=val_data_gen.steps,
    #                     epochs=n_trained_chunks + args.epochs,
    #                     initial_epoch=n_trained_chunks,
    #                     callbacks=[metrics_callback, saver, csv_logger],
    #                     verbose=args.verbose)
    # import cProfile
    # cProfile.run('history = model.fit(x=train_data_gen, steps_per_epoch=train_data_gen.steps, epochs=n_trained_chunks + args.epochs, initial_epoch=n_trained_chunks, callbacks=[metrics_callback, saver, csv_logger], verbose=args.verbose, workers=1, use_multiprocessing=False)')

    history = model.fit(x=train_data_gen,
            steps_per_epoch=train_data_gen.steps,
            #validation_data=val_data_gen,
            #validation_steps=val_data_gen.steps,
            epochs=n_trained_chunks + args.epochs,
            initial_epoch=n_trained_chunks,
            callbacks=[metrics_callback, saver, csv_logger, earlyStopping],
            verbose=args.verbose,
            workers=args.workers,
            use_multiprocessing=True)
    print("==> /training")

    discretizer.print_statistics()
    #aflanders: end

elif args.mode == 'test':
    # ensure that the code uses test_reader
    # del train_data_gen
    # del val_data_gen

    names = []
    ts = []
    labels = []
    predictions = []

    if args.deep_supervision:
        # del train_data_loader
        # del val_data_loader
        test_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'test'),
                                                                  listfile=os.path.join(args.data, 'test_listfile.csv'),
                                                                  small_part=args.small_part)
        test_data_gen = utils.BatchGenDeepSupervision(test_data_loader, args.partition,
                                                      discretizer, normalizer, args.batch_size,
                                                      shuffle=False, return_names=True)
        for i in range(test_data_gen.steps):
            print("\tdone {}/{}".format(i, test_data_gen.steps), end='\r')

            ret = test_data_gen.next(return_y_true=True)
            (x, y_processed, y) = ret["data"]
            cur_names = np.array(ret["names"]).repeat(x[0].shape[1], axis=-1)
            cur_ts = ret["ts"]
            for single_ts in cur_ts:
                ts += single_ts

            pred = model.predict(x, batch_size=args.batch_size)
            if pred.shape[-1] == 1:  # regression
                pred_flatten = pred.flatten()
            else:  # classification
                pred_flatten = pred.reshape((-1, 10))
            for m, t, p, name in zip(x[1].flatten(), y.flatten(), pred_flatten, cur_names.flatten()):
                if np.equal(m, 1):
                    labels.append(t)
                    predictions.append(p)
                    names.append(name)
    else:
        # del train_reader
        # del val_reader
        test_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'test'),
                                         listfile=os.path.join(args.data, 'test_listfile.csv'))
        test_data_gen = utils.BatchGen(reader=test_reader,
                                       discretizer=discretizer,
                                       normalizer=normalizer,
                                       partition=args.partition,
                                       batch_size=args.batch_size,
                                       steps=args.val_batches,  # put steps = None for a full test
                                       shuffle=False,
                                       verbose=args.verbose)

        y_true = []
        predictions = []

        # for i in tqdm(range(test_data_gen.steps), desc='Predicting batches'):
        #     print("predicting {} / {}".format(i, test_data_gen.steps), end='\r')
        pred = model.predict(test_data_gen, 
                            batch_size=args.batch_size, 
                            verbose=args.verbose, 
                            steps=args.val_batches,
                            workers=args.workers,
                            use_multiprocessing=True)

        names, ts, y = zip(*test_data_gen.get_truth(len(pred)))
        names = list(names)
        ts = list(ts)

        # y = test_data_gen.get_y(len(pred))
        labels += list(y)
        predictions += list(pred)

            # ret = test_data_gen.next(return_y_true=True)
            # (x, y_processed, y) = ret["data"]
            # cur_names = ret["names"]
            # cur_ts = ret["ts"]

            # x = np.array(x)
            # pred = model.predict_on_batch(x)
            # predictions += list(pred)
            # labels += list(y)
            # names += list(cur_names)
            # ts += list(cur_ts)

    if args.partition == 'log':
        predictions = [metrics.get_estimate_log(x, 10) for x in predictions]
        metrics.print_metrics_log_bins(labels, predictions)
    if args.partition == 'custom':
        predictions = [metrics.get_estimate_custom(x, 10) for x in predictions]
        metrics.print_metrics_custom_bins(labels, predictions)
    if args.partition == 'none':
        metrics.print_metrics_regression(labels, predictions)
        predictions = [x[0] for x in predictions]

    path = os.path.join(os.path.join(args.output_dir, "test_predictions", os.path.basename(args.load_state)) + ".csv")
    utils.save_results(names, ts, predictions, labels, path)

else:
    raise ValueError("Wrong value for args.mode")
