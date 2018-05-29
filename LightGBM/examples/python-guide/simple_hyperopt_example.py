# from sklearn import datasets
# from sklearn.cross_validation import train_test_split
# import xgboost as xgb
import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK


# load or create your dataset
print('Load data...')
df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')

y_train = df_train[0].values
y_test = df_test[0].values
X_train = df_train.drop(0, axis=1).values
X_test = df_test.drop(0, axis=1).values

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


def GBM(argsDict):
    #读取参数
    # num_leaves = argsDict['num_leaves']
    learning_rate = argsDict['learning_rate']
    feature_fraction = argsDict['feature_fraction']
    bagging_fraction = argsDict['bagging_fraction']
    bagging_freq = argsDict['bagging_freq']

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'auc'},
        'num_leaves': 31,
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'verbose': 0
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5)

    # print('Save model...')
    # save model to file
    # gbm.save_model('model.txt')

    # print('Start predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    return  mean_squared_error(y_test, y_pred) ** 0.5


space = {
         "learning_rate":hp.uniform("learning_rate",0.01,1),  #[0,1,2,3,4,5] -> [50,]
         "feature_fraction":hp.uniform("feature_fraction",0.01,0.99),  #[0,1,2,3,4,5] -> 0.05,0.06
         "bagging_fraction":hp.uniform("bagging_fraction",0.25,0.95),#[0,1,2,3] -> [0.7,0.8,0.9,1.0]
         "bagging_freq":hp.randint("bagging_freq",10), #
        }
algo = partial(tpe.suggest,n_startup_jobs=1)
best = fmin(GBM,space,algo=algo,max_evals=4)

print('best:'+str(best))
print('bgmBest'+str(GBM(best)))