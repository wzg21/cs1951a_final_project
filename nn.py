import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold

import statsmodels.api as sm
from statsmodels.tools import eval_measures

from sklearn.tree import DecisionTreeRegressor

# all transaction types
all_trans_types = ['delivery', 'pickup', 'restaurant_reservation']

# all 23 categories 
all_cats = ['italian', 'breakfast_brunch', 'newamerican', 'pizza', 'tradamerican', 'sandwiches', 'chinese', 'bars', 'japanese', 'mexican',
'seafood', 'delis', 'coffee', 'salad', 'mediterranean', 'french', 'indpak', 'thai', 'asianfusion', 'korean', 'latin', 'desserts', 'bakeries']

# all 23 category ratios 
all_cats_ratios = [c + "_ratio" for c in all_cats]

# non-category variables 
all_non_cats = all_trans_types + ['latitude', 'longitude', 'price']

# all 52 independent variables
all_ind_vars = all_non_cats + all_cats + all_cats_ratios

# read all variables into a dataframe
all_variables = pd.read_csv('./data/businesses_nyc_small.csv', usecols=['rating'] + all_ind_vars)
labels = all_variables.pop('rating')

def build_model(size):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[size]),
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


def MLR(train, test, train_label, test_label):
    train_ = sm.add_constant(train)
    res = sm.OLS(train_label, train_).fit()
    pred_train = res.predict(train_)
    test = sm.add_constant(test)
    pred_test = res.predict(test)
    mlr_test_mse = eval_measures.mse(test_label, pred_test)
    mlr_train_mse = eval_measures.mse(train_label, pred_train)
    return mlr_train_mse, mlr_test_mse


def cross_validation(all_variables, labels, ind_variables, title):
    
    features = all_variables[ind_variables].to_numpy()
    labels = labels.to_numpy()

    k_fold = KFold(n_splits=5, random_state=0, shuffle=True)
    nn_test_mse_sum = nn_train_mse_sum = mlr_test_mse_sum = mlr_train_mse_sum = dtr_train_mse_sum = dtr_test_mse_sum = base_test_mse_sum = base_train_mse_sum = 0
    for train_indices, test_indices in k_fold.split(features):
        train = features[train_indices]
        train_label = labels[train_indices]
        test = features[test_indices]
        test_label = labels[test_indices]

        # neural network
        epoch = 10
        model = build_model(features.shape[1])
        history = model.fit(
            train, train_label,
            epochs=epoch, verbose=0)
        # plot_history(history)
        loss, mae, nn_test_mse = model.evaluate(test, test_label, verbose=2)
        nn_train_mse = history.history['mse'][epoch-1]
        print('nn test mse: ' + str(nn_test_mse))
        print('nn train mse: ' + str(nn_train_mse))
        nn_test_mse_sum += nn_test_mse
        nn_train_mse_sum += nn_train_mse

        # multiple linear regression
        mlr_train_mse, mlr_test_mse = MLR(train, test, train_label, test_label)
        print('mlr test MSE: ' + str(mlr_test_mse))
        print('mlr train mse: ' + str(mlr_train_mse))
        mlr_test_mse_sum += mlr_test_mse
        mlr_train_mse_sum += mlr_train_mse

        # decision tree regression
        regr = DecisionTreeRegressor(max_depth=5)
        regr.fit(train, train_label)
        target_train = regr.predict(train)
        target_test = regr.predict(test)
        regression_train_mse = eval_measures.mse(train_label, target_train)
        print("Decision Tree Regression train MSE: ", regression_train_mse)
        regression_test_mse = eval_measures.mse(test_label, target_test)
        print("Decision Tree Regression test MSE: ", regression_test_mse)
        dtr_train_mse_sum += regression_train_mse
        dtr_test_mse_sum += regression_test_mse

        # baseline
        mean = np.mean(list(train_label) + list(test_label))
        base_test_mse = sum([(label - mean)**2 for label in test_label])/len(test_label)
        print('baseline (mean) test mse: ' + str(base_test_mse))
        base_train_mse = sum([(label - mean)**2 for label in train_label])/len(train_label)
        print('baseline (mean) train mse: ' + str(base_train_mse))
        base_test_mse_sum += base_test_mse
        base_train_mse_sum += base_train_mse

    df = pd.DataFrame([])
    df['model'] = ['MLR', 'MLR', 'NN', 'NN', 'Decision Tree', 'Decision Tree', 'baseline', 'baseline']
    df['data'] = ['train', 'test', 'train', 'test', 'train', 'test', 'train', 'test']
    df['MSE'] = [mlr_train_mse_sum/5, mlr_test_mse_sum/5, nn_train_mse_sum/5, nn_test_mse_sum/5, dtr_train_mse_sum/5, dtr_test_mse_sum/5, base_train_mse_sum/5, base_test_mse_sum/5]
    sns.barplot(x="model", y="MSE", hue="data", data=df, palette="Paired")
    plt.title(title)
    plt.show()

# full model - all 52 features
cross_validation(all_variables, labels, all_ind_vars, 'Model results with all features')

# # smallest model - non-category features (transactions + longitude/latitude + price)
# cross_validation(all_variables, labels, all_non_cats)

# # all features except catgory ratios
# # compare this with the 1st model to see if having category ratios helps the prediction
# # compare this with the 2nd model to see if having categories helps the prediction
# cross_validation(all_variables, labels, all_non_cats + all_cats)

# without transaction
cross_validation(all_variables, labels, ['latitude', 'longitude', 'price']+all_cats+all_cats_ratios, 'Model results without transaction type features')

