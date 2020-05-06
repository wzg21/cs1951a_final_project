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
    res = []
    for train_indices, test_indices in k_fold.split(features):
        train = features[train_indices]
        train_label = labels[train_indices]
        test = features[test_indices]
        test_label = labels[test_indices]

        # neural network
        epoch = 200
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

        res.extend([mlr_train_mse, mlr_test_mse, regression_train_mse, regression_test_mse, nn_train_mse, nn_test_mse, base_train_mse, base_test_mse])

    df = pd.DataFrame([])
    df['Model'] = ['MLR', 'MLR', 'DT', 'DT', 'NN', 'NN', 'Baseline', 'Baseline']*5
    df['data'] = ['Train', 'Test', 'Train', 'Test', 'Train', 'Test', 'Train', 'Test']*5
    df['MSE'] = res #[mlr_train_mse_sum/5, mlr_test_mse_sum/5, dtr_train_mse_sum/5, dtr_test_mse_sum/5, nn_train_mse_sum/5, nn_test_mse_sum/5, base_train_mse_sum/5, base_test_mse_sum/5]
    sns.barplot(x="Model", y="MSE", hue="data", data=df, palette="Paired")
    plt.title(title)
    plt.legend(loc=[1, 1])
    plt.tight_layout()
    #plt.box(False)
    plt.show()
    plt.savefig(fname="D:\Brown_MS\spring2020\cs1951a\cs1951a_final_project\pictures\models\\"+title,format="svg")
    mlr, nn, dt, base = mlr_test_mse_sum/5, nn_test_mse_sum/5, dtr_test_mse_sum/5, base_test_mse_sum/5
    return mlr, nn, dt, base, res

# full model - all 52 features
mlr1, nn1, dt1, base1, wf = cross_validation(all_variables, labels, all_ind_vars, 'Full model')

mlr2, nn2, dt2, base2, _ = cross_validation(all_variables, labels, all_cats_ratios + ['price', 'latitude', 'longitude'] + all_trans_types, 'Full model w/o category features')

mlr3, nn3, dt3, base3, wr = cross_validation(all_variables, labels, all_cats + ['price', 'latitude', 'longitude'] + all_trans_types, 'Full model w/o category ratio features')

mlr4, nn4, dt4, base4, _ = cross_validation(all_variables, labels, all_cats + ['latitude', 'longitude'] + all_trans_types + all_cats_ratios, 'Full w/o without price feature')

mlr5, nn5, dt5, base5, wl = cross_validation(all_variables, labels, all_cats + all_trans_types + all_cats_ratios + ['price'], 'Full model w/o location features')

mlr6, nn6, dt6, base6, _ = cross_validation(all_variables, labels, all_cats + ['price', 'latitude', 'longitude'] + all_cats_ratios, 'Full model w/o transaction type features')

df = pd.DataFrame([])
df['Model'] = ['MLR', 'MLR', 'DT', 'DT', 'NN', 'NN', 'Baseline', 'Baseline']*15
df['data'] = ['Full model']*40 + ['W/O category ratio features']*40 + ['W/O location features']*40
df['MSE'] = wf + wr + wl
sns.barplot(x="Model", y="MSE", hue="data", data=df, palette="Paired")
plt.title('w/o category ratio VS w/o location')
plt.legend(loc=[1, 1])
plt.tight_layout()
#plt.box(False)
plt.show()
plt.savefig(fname="D:\Brown_MS\spring2020\cs1951a\cs1951a_final_project\pictures\models\claim3",format="svg")

table = [[mlr1, dt1, nn1, base1],
[mlr2, dt2, nn2, base2],
[mlr3, dt3, nn3, base3],
[mlr4, dt4, nn4, base4],
[mlr5, dt5, nn5, base5],
[mlr6, dt6, nn6, base6]]
table = np.transpose(table)
print(table)

# data = pd.DataFrame(data=table, columns=['all features','without category features', 
#                                         'category ratio','price',
#                                        'location', 'transaction types'],
#                    index=['MLR','NN','DT','Baseline'])

table = [[mlr3, dt3, nn3, base3],
[mlr5, dt5, nn5, base5]]
print(table)

# data = pd.DataFrame(data=table, index=['all features except category ratio','all features except location'],
#                    columns=['MLR','NN','DT','Baseline'])

