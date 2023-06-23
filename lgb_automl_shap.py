import warnings
import shap
import math
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from hyperopt import fmin, hp, Trials, space_eval, rand, type, anneal
from sklearn.metrics import roc_auc_score, mean_absolute_percentage_error, log_loss
import lightgbm as lgb

class auto_explain:
    def __init__(self, train, res_column, test = None, test_size = None):
        # 如果有指定test test
        if test:
            self.train_x, self.train_y = train.drop(res_column, axis = 1), train.res_column
            self.test_x, self.test_y = test.drop(res_column, axis = 1), test.res_column
            self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.train_x, self.train_y, test_size = 0.1, random_state=0)
        # 如果没有，就从train里切train,test,val(val可有可无，可自行删除，默认7-2-1分布)
        else:
            x = train.drop(res_column, axis = 1)
            y = train.res_column
            if test_size:
                self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, test_size = test_size, random_state=0)
                self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.train_x, self.train_y, test_size = 0.13, random_state = 0)
            else:
                self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, test_size = 0.2, random_state=0)
                self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.train_x, self.train_y, test_size = 0.13, random_state = 0) 
        self.gbm, self.explainer, self.shap_matrix = None, None, None
        self.lgb_train, self.lgb_test, self.lgb_valid = lgb.Dataset(self.train_x, self.train_y), lgb.Dataset(self.test_x, self.test_y), lgb.Dataset(self.val_x, self.val_y)
        self.x = pd.concat([self.train_x, self.test_x, self.val_x])
        self.y = pd.concat([self.train_y, self.test_y, self.val_y])
    
    def tune_gbm(self, mode, eval_func = None, eval_res_func = None, params = None, spaces = None, max_evals = 50, boost_round = 300, early_stop_rounds = 30):
        def auc(real, pred):
            return roc_auc_score(real, pred)
        
        def mape(real, pred):
            return mean_absolute_percentage_error(real, pred)
        
        def logloss(real, pred):
            return log_loss(real, pred)
        
        def multi_logloss(real, pred):
            return np.mean([-math.log(pred[i][y]) for i, y in enumerate(real)])
        
        if mode == 'binary' and not params:
            eval_func = logloss
            if not eval_res_func:
                eval_res_func = auc
            
        if mode == 'regression' and not params:
            mode = 'mape'
            if not eval_res_func:
                eval_func = mape
        
        if mode == 'multiclass' and not params:
            mode = 'multiclass'
            eval_func = multi_logloss
            if not eval_res_func:
                eval_res_func = multi_logloss
        
        if not params:
            params = {
                'objective': None,
                'max_depth': 6,
                'num_leaves': 50,
                'learning_rate': 0.1,
                'num_iterations': 300,
                'colsample_bytree': 0.8,
                'subsample': 0.8,
                'n_jobs': -1,
                'feature_pre_filter': False,
                'verbose': -1,
                'num_class': None if mode != 'multiclass' else len(np.unique(self.train_y))
            }

        if not spaces:
            spaces = {
                'objective': mode,
                'feature_pre_filter': False,
                'verbose': -1,
                'n_jobs': -1,
                'num_class': None if mode != 'multiclass' else len(np.unique(self.train_y)),
                # 上面的在early stopping里有用，别删
                'bagging_fraction': np.uniform('bagging_function', 0.65, 0.8),
                
            }
                    

