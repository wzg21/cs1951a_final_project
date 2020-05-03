import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold

import statsmodels.api as sm
from statsmodels.tools import eval_measures

features = pd.read_csv('businesses_nyc_small.csv', usecols=['rating', 'latitude', 'longitude', 'delivery', 'pickup', 'restaurant_reservation', 'price',
		'italian', 'breakfast_brunch', 'newamerican', 'pizza', 'tradamerican', 'sandwiches', 'chinese', 'bars', 'japanese', 'mexican',
		'seafood', 'delis', 'coffee', 'salad', 'mediterranean', 'french', 'indpak', 'thai', 'asianfusion', 'korean', 'latin', 'desserts', 'bakeries'])
labels = features.pop('rating')
features = features.to_numpy()
labels = labels.to_numpy()

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[features.shape[1]]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
    return model

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    # plt.figure()
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean Abs Error')
    # plt.plot(hist['epoch'], hist['mae'],
    #         label='Train Error')
    # plt.ylim([0,2])
    # plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mse'],
            label='Train Error')
    plt.ylim([0,3])
    plt.legend()
    plt.show()

k_fold = KFold(n_splits=5, random_state=0, shuffle=True)
for train_indices, test_indices in k_fold.split(features):
    train = features[train_indices]
    train_label = labels[train_indices]
    test = features[test_indices]
    test_label = labels[test_indices]

    # neural network
    epoch = 200
    model = build_model()
    history = model.fit(
        train, train_label,
        epochs=epoch, verbose=0)
    # plot_history(history)
    loss, mae, nn_test_mse = model.evaluate(test, test_label, verbose=2)
    nn_train_mse = history.history['mse'][epoch-1]
    print('nn test mse: ' + str(nn_test_mse))
    print('nn train mse: ' + str(nn_train_mse))

    # multiple linear regression
    train_ = sm.add_constant(train)
    res = sm.OLS(train_label, train_).fit()
    pred_train = res.predict(train_)
    test = sm.add_constant(test)
    pred_test = res.predict(test)
    mlr_test_mse = eval_measures.mse(test_label, pred_test)
    mlr_train_mse = eval_measures.mse(train_label, pred_train)
    print('mlr test MSE: ' + str(mlr_test_mse))
    print('mlr train mse: ' + str(mlr_train_mse))

    # baseline
    mean = np.mean(list(train_label) + list(test_label))
    base_test_mse = sum([(label - mean)**2 for label in test_label])/len(test_label)
    print('baseline (mean) test mse: ' + str(base_test_mse))
    base_train_mse = sum([(label - mean)**2 for label in train_label])/len(train_label)
    print('baseline (mean) train mse: ' + str(base_train_mse))

    df = pd.DataFrame([])
    df['model'] = ['MLR', 'MLR', 'NN', 'NN', 'baseline', 'baseline']
    df['data'] = ['train', 'test', 'train', 'test', 'train', 'test']
    df['MSE'] = [mlr_train_mse, mlr_test_mse, nn_train_mse, nn_test_mse, base_train_mse, base_test_mse]
    sns.barplot(x="model", y="MSE", hue="data", data=df, palette="Paired")
    plt.show()

