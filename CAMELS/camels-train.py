import os
import numpy as np
import pandas as pd

from multicnn import MultiCNN

import tensorflow as tf
from tensorflow.keras import models as km
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks as kc
from tensorflow.keras import optimizers as ko

from sklearn.model_selection import train_test_split


def load_data(f : str):
    assert f is not None, "Target datafile should be specified in `load_data()`!"

    # Load a selected datafile
    X = np.load(os.path.join(DDIR, f), allow_pickle=True)
    X = np.log10(X)

    # Load labels
    dataset = '_'.join(f.split('_')[2:-2])
    y = np.genfromtxt(os.path.join(DDIR, f"params_{dataset}.txt"))
    y = np.repeat(y, 15, axis=0)

    test_size = 0.33
    valid_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                            test_size=test_size, random_state=57)
    del(X)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                            test_size=valid_size/(1-test_size), random_state=57)

    #
    # TENSORFLOW PURGATORY IN EARLY 2022
    #
    # Wrap data in Dataset objects
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    valid_data = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    del(X_train); del(y_train)
    del(X_valid); del(y_valid)
    del(X_test); del(y_test)

    # The batch size must now be set on the Dataset objects
    batch_size = 128
    train_data = train_data.batch(batch_size)
    valid_data = valid_data.batch(batch_size)

    # Disable AUTO sharding policy
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
                            tf.data.experimental.AutoShardPolicy.OFF
    train_data = train_data.with_options(options)
    valid_data = valid_data.with_options(options)
    test_data = test_data.with_options(options)

    return train_data, valid_data, test_data


def prepare_model(gpu: str = '0'):
    GPU = [f"GPU:{i}" for i in gpu.split(',')]

    if len(gpu.split(',')) > 1:
        strategy = tf.distribute.MirroredStrategy(GPU)
    else:
        strategy = tf.distribute.OneDeviceStrategy(GPU[0])

    with strategy.scope():
        # Initialize the CNN template for all network branches
        multi_cnn = MultiCNN(
            imsize=256,
            n_channels=1,
            num_filters=16,
            kernelsize=3,
            padding='same',
            stride=1,
            kreg=5e-05,
            activation='relu'
        )

        # Add branches for all target value
        multi_cnn.add_branch(n_target=1, branch_name="Omega_m")
        multi_cnn.add_branch(n_target=1, branch_name="sigma_8")
        multi_cnn.add_branch(n_target=1, branch_name="A_SN1")
        multi_cnn.add_branch(n_target=1, branch_name="A_AGN1")
        multi_cnn.add_branch(n_target=1, branch_name="A_SN2")
        multi_cnn.add_branch(n_target=1, branch_name="A_AGN2")

        # Compile the model 
        model = multi_cnn.get_model()
        model.compile(loss='mean_squared_error',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

        # Create callback checkpoint
        best_model = kc.ModelCheckpoint('best_model.hdf5',
                                    save_best_only=True, verbose=1)

    return model, best_model
      

def main():
    DDIR = '/mnt/local/scratch-sata-stripe/scratch/masterdesky/data/CAMELS/2D_maps/data/'
    FILE = os.listdir(DDIR)
    FILE = sorted([f for f in FILE if ('.txt' not in f) & ('_CV_' not in f)])

    train_data, valid_data, test_data = load_data(f=FILE[18])
    model, best_model = prepare_model(gpu='0,1,2')
    
    # Train the model
    epochs = 5
    #batch_size = 128
    history = model.fit(train_data,#x=X_train, y=y_train,
                        validation_data=valid_data,#(X_valid, y_valid),
                        epochs=epochs,#batch_size=batch_size,
                        callbacks=[best_model])


if __name__ == "main":
    main()