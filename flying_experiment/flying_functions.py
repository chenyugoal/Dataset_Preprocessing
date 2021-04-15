import keras
import numpy as np
from keras import layers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from proglearn.deciders import SimpleArgmaxAverage
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.transformers import NeuralClassificationTransformer, TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter

from tensorflow.keras.backend import clear_session  # To avoid OOM error when using dnn


def single_experiment(train_x_task, test_x_task, train_y_task, test_y_task, ntrees=10, model='uf'):
    num_tasks = 10
    num_points_per_task = 1800
    accuracies = np.zeros(65, dtype=float)

    if model == 'dnn':

        clear_session()  # clear GPU memory before each run, to avoid OOM error

        default_transformer_class = NeuralClassificationTransformer

        network = keras.Sequential()
        network.add(
            layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=np.shape(train_x_task[0])[1:]))
        network.add(layers.BatchNormalization())
        network.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding="same", activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding="same", activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding="same", activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Conv2D(filters=254, kernel_size=(3, 3), strides=2, padding="same", activation='relu'))

        network.add(layers.Flatten())
        network.add(layers.BatchNormalization())
        network.add(layers.Dense(2000, activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Dense(2000, activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Dense(units=20, activation='softmax')) # units=10

        default_transformer_kwargs = {
            "network": network,
            "euclidean_layer_idx": -2,
            "loss": "categorical_crossentropy",
            "optimizer": Adam(3e-4),
            "fit_kwargs": {
                "epochs": 100,
                "callbacks": [EarlyStopping(patience=5, monitor="val_loss")],
                "verbose": False,
                "validation_split": 0.33,
                "batch_size": 32,
            },
        }
        default_voter_class = KNNClassificationVoter
        default_voter_kwargs = {"k": int(np.log2(num_points_per_task))}
        default_decider_class = SimpleArgmaxAverage

    elif model == 'uf':
        for i in range(num_tasks):
            train_x_task[i] = train_x_task[i].reshape(1080, -1)
            test_x_task[i] = test_x_task[i].reshape(720, -1)

        default_transformer_class = TreeClassificationTransformer
        default_transformer_kwargs = {"kwargs": {"max_depth": 30}}
        default_voter_class = TreeClassificationVoter
        default_voter_kwargs = {}
        default_decider_class = SimpleArgmaxAverage

    progressive_learner = ProgressiveLearner(default_transformer_class=default_transformer_class,
                                             default_transformer_kwargs=default_transformer_kwargs,
                                             default_voter_class=default_voter_class,
                                             default_voter_kwargs=default_voter_kwargs,
                                             default_decider_class=default_decider_class)

    for i in range(num_tasks):
        progressive_learner.add_task(
            X=train_x_task[i],
            y=train_y_task[i],
            task_id=i,
            num_transformers=1 if model == "dnn" else ntrees,
            transformer_voter_decider_split=[0.67, 0.33, 0],
            decider_kwargs={"classes": np.unique(train_y_task[i])},
        )
        prediction = progressive_learner.predict(
            X=test_x_task[i], transformer_ids=[i], task_id=i
        )
        accuracies[i] = np.mean(prediction == test_y_task[i])

        for j in range(num_tasks):
            if j > i:
                pass  # this is not wrong but misleading, should be continue
            else:
                odif_predictions = progressive_learner.predict(test_x_task[j], task_id=j)

            accuracies[10 + j + (i * (i + 1)) // 2] = np.mean(odif_predictions == test_y_task[j])
    print('single experiment done!')

    return accuracies


def calculate_results(accuracy_all):
    num_tasks = 10
    err = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            err[i].append(1 - accuracy_all[10 + ((j + i) * (j + i + 1)) // 2 + i])

    bte = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            bte[i].append(err[i][0] / err[i][j])

    fte = np.zeros(10, dtype=float)
    for i in range(num_tasks):
        fte[i] = (1 - accuracy_all[i]) / err[i][0]

    te = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            te[i].append((1 - accuracy_all[i]) / err[i][j])

    return err, bte, fte, te
