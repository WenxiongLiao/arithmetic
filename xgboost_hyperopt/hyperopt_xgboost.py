from sklearn import datasets
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn import metrics
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK

iris = datasets.load_iris()
data = iris.data[:100]
print(data.shape)
#(100L, 4L)
#一共有100个样本数据, 维度为4维

label = iris.target[:100]
print(label)

train_x, test_x, train_y, test_y = train_test_split(data, label, random_state=0)
dtrain=xgb.DMatrix(train_x,label=train_y)
dtest=xgb.DMatrix(test_x)


def GBM(argsDict):
    #读取参数
    max_depth = argsDict['max_depth']
    lambda1 = argsDict['lambda']
    subsample = argsDict['subsample']
    colsample_bytree = argsDict['colsample_bytree']
    min_child_weight = argsDict['min_child_weight']
    eta = argsDict['eta']

    #设置参数
    params={'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth':max_depth,
        'lambda':lambda1,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight,
        'eta': eta,
        'seed':0,
        'nthread':8,
         'silent':1}

    watchlist = [(dtrain,'train')]

    #训练
    bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)

    ypred=bst.predict(dtest)

    # 设置阈值, 输出一些评价指标
    y_pred = (ypred >= 0.5)*1


    # print('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))
    # print('ACC: %.4f' % metrics.accuracy_score(test_y,y_pred))
    # print('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
    # print('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
    # print('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
    metrics.confusion_matrix(test_y,y_pred)
    #把f1值返回
    return -metrics.f1_score(test_y,y_pred)


space = {"max_depth":hp.randint("max_depth",15),
         "lambda":hp.randint("lambda",10),  #[0,1,2,3,4,5] -> [50,]
         "colsample_bytree":hp.uniform("colsample_bytree",0.01,0.9),  #[0,1,2,3,4,5] -> 0.05,0.06
         "subsample":hp.uniform("subsample",0.25,0.75),#[0,1,2,3] -> [0.7,0.8,0.9,1.0]
         "min_child_weight":hp.randint("min_child_weight",5), #
         "eta": hp.uniform("eta", 0.01,0.5),  #
        }
algo = partial(tpe.suggest,n_startup_jobs=1)
best = fmin(GBM,space,algo=algo,max_evals=4)

print('best:'+str(best))
print('bgmBest'+str(GBM(best)))