import os
import time
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from IPython.display import display

# Base classes
from sklearn.base import ClassifierMixin, TransformerMixin

# Random search & splitting
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier

# Transformers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import FunctionTransformer

# Metrics
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score

# transformers
from twitter_utils import twitter_feature_generation_transformer, twitter_feature_selection_transformer

# custom scorer
def afr(y_true, y_pred, **kwargs):
    """The afr scoring function gives a weighted average of accuracy,
    balanced F1 and roc_auc scores, with .2, .3 and .5 respectively.
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_true, y_pred, average="weighted")
    return (accuracy * .2) + (f1 * .3) + (roc_auc * .5)

afr_scorer = make_scorer(afr, greater_is_better=True)
afr_scorer.name = 'afr'

def passthrough(df):
    return df.copy()

passthrough_transformer = FunctionTransformer(passthrough)

# Base class for pipeline 
class BinaryClassifierPipelineStep(TransformerMixin, ClassifierMixin):
    def __init__(self, context={}):
        self._context = context
    
    def set_context(self, context):
        self._context = context
        return self
    
    def get_fitted_classifiers(self):
        return self._context._clfs
    
    def get_best_classifier(self):
        return self._context._best
    
    def get_info(self):
        return self._context._info

class DataSplitStep(BinaryClassifierPipelineStep):
    """
    """
    def __init__(self, training_size=.64, validation_size=.16, splitting=None):        
        self._training = training_size 
        self._validation = validation_size
        
        if callable(splitting):
            self._splitter = splitting
        else:
            self._splitter = train_test_split
    
    def transform(self, X, y):
        t_size = self._training if isinstance(self._training, int) else int(len(X)*self._training)        
        X_tv, X_train, y_tv, y_train = self._splitter(X, y, test_size=t_size, random_state=42)
        
        # Split remaing data into training and validation
        t_size = self._validation if isinstance(self._validation, int) else int(len(X)*self._validation)        
        X_val, X_test, y_val, y_test = self._splitter(X_tv, y_tv, test_size=t_size, random_state=42)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

class FeatureEngineeringStep(BinaryClassifierPipelineStep):
    """
    """
    def __init__(self, feature_generation=None, feature_selection=None, feature_scaling=None, context={}):
        self._context = context
        
        # Define the trasnformer to be used for feature generation
        if callable(getattr(feature_generation, "transform", None)):
            self._feature_generation = feature_generation
        else:
            self._feature_generation = twitter_feature_generation_transformer
        
        # Define the trasnformer to be used for feature selection
        if callable(getattr(feature_selection, "transform", None)):
            self._feature_selection = feature_selection
        else:
            self._feature_selection = twitter_feature_selection_transformer
        
        # The transformer used for feature scaling will be defined after the 
        # pipeline context is set
        self._scaling = feature_scaling
    
    def _setup(self):
        # Define the transformer to be used for feature scaling
        self._context._scaling = self._scaling
        if self._scaling == 'min-max':
            self._context._scaler = MinMaxScaler()
        elif self._scaling == 'standard':
            self._context._scaler = StandardScaler()
        elif callable(getattr(self._scaling, "transform", None)):
            self._context._scaling = getattr(self._scaling, "name", type(self._scaling))
            self._context._scaler = self._scaling
        else:
            self._context._scaling = 'none'
            self._context._scaler = passthrough_transformer
    
    def fit_transform(self, X):
        # Setup scaler on the pipeline context
        self._setup()
        
        # Run feature generation transform
        X_t = self._feature_generation.transform(X)
        
        # Run feature selection transform
        X_t = self._feature_selection.transform(X_t)
                
        # run feature scaling transform
        return self._context._scaler.fit_transform(X_t)
    
    def transform(self, X):
        X_t = self._feature_generation.transform(X)        
        X_t = self._feature_selection.transform(X_t)
        return self._context._scaler.transform(X_t)

class ModelTuningStep(BinaryClassifierPipelineStep):
    def __init__(self, scoring=None, n_iter=20):        
        
        # Define number of iteractions for random search
        self._n_iter = n_iter
        
        # the scoring will be setup after context is set
        self._scoring = scoring
    
    # Setup must run after pipeline context is set
    def _setup(self):
        # Define the scoring function used in cross-validation
        self._context._scoring = self._scoring
        if self._scoring == 'roc_auc':
            self._context._scorer = make_scorer(roc_auc_score)
        elif self._scoring == 'f1':
            self._context._scorer = make_scorer(f1_score)
        elif self._scoring == 'accuracy':
            self._context._scorer = make_scorer(accuracy_score)
        elif callable(self._scoring):
            self._context._scoring = getattr(self._scoring, "name", type(self._scoring))
            self._context._scorer = self._scoring
        else:
            self._context._scoring = 'roc_auc'
            self._context._scorer = make_scorer(roc_auc_score)
        
        # Define the classifier's hyper-parameters search space used in 
        # the training and tuning step
        self._context._clfs = {
            'knn': {
                'name': 'K-Nearest Neighbors',
                'base': KNeighborsClassifier(),
                'param_distributions': {
                    'n_neighbors': [int(x) for x in np.linspace(2, 50, num=20)],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
            },
            'log_reg': {
                'name': 'Logistic Regression',
                'base': LogisticRegression(random_state=42),
                'param_distributions': {
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'fit_intercept': [True, False],
                    'max_iter': [int(x) for x in np.linspace(100, 1000, num=20)]
                }
            },
            'svm': {
                'name': 'Support Vector Machines',
                'base': SVC(random_state=42),
                'param_distributions': {
                    'C': [round(x, 3) for x in np.linspace(0.1, 5, num=10)],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto']
                }
            },
            'tree': {
                'name': 'Decision Tree',
                'base': DecisionTreeClassifier(random_state=42),
                'param_distributions': {
                    "max_depth": [int(x) for x in np.linspace(1, 12, num=5)],
                    "max_features": [int(x) for x in np.linspace(1, 20, num=5)],
                    "min_samples_leaf": [int(x) for x in np.linspace(1, 200, num=20)],
                    "criterion": ["gini", "entropy"]
                }
            },
            'rf': {
                'name': 'Random Forest',
                'base': RandomForestClassifier(random_state=42),
                'param_distributions': {
                    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1500, num=10)],
                    'max_features': [.5, 'sqrt', 'log2'],
                    'max_depth': [None] + [int(x) for x in np.linspace(5, 50, num=10)],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                }
            },
            'bagging': {
                'name': 'Baggin',
                'base': BaggingClassifier(random_state=42),
                'param_distributions': {
                    "n_estimators": [int(x) for x in np.linspace(start=100, stop=1500, num=10)],
                    "max_samples": np.arange(start=0.1, stop=1.0, step=0.1)
                }
            },
            'xgboost': {
                'name': 'XGBoost',
                'base': XGBClassifier(random_state=42, objective='binary:logistic'),
                'param_distributions': {
                    'n_estimators': stats.randint(150, 1000),
                    'learning_rate': stats.uniform(0.01, 0.6),
                    'max_depth': [3, 4, 5, 6, 7, 8, 9]
                }
            }
        }
        return self
        
    def fit(self, X_training, y_training):
        # setup context objects
        self._setup()
        self._context._X_training = X_training        
        self._context._y_training = y_training
        
        for clf in self._context._clfs:
            # Load a previously fitted model
            if self._context._save and os.path.isfile(f'output/models/{self._context._name}/{clf}.model'):
                self._context._clfs[clf]['sclf'] = pickle.load(open(f'output/models/{self._context._name}/{clf}.model', 'rb'))
            # or fit the model and save it for future use
            else:
                t = time.process_time()
                sclf = RandomizedSearchCV(
                    self._context._clfs[clf]['base'],
                    self._context._clfs[clf]['param_distributions'],
                    random_state=42,
                    n_iter=self._n_iter,
                    cv=None,#self._cv,
                    scoring=self._context._scorer,
                    n_jobs=-1
                ).fit(X_training, y_training)
                sclf.run_time = time.process_time() - t
                
                if self._context._save:
                    if not os.path.exists(f'output/models/{self._context._name}'):
                        os.makedirs(f'output/models/{self._context._name}')
                    pickle.dump(sclf, open(f'output/models/{self._context._name}/{clf}.model', 'wb'))
                
                self._context._clfs[clf]['sclf'] = sclf
        return self
                    
class ModelSelectionStep(BinaryClassifierPipelineStep):
    def __init__(self, scoring=None):        
        self._scoring_p = scoring
    
    def _setup(self):
        # Define the scoring function used in cross-validation and model selection
        self._scoring = self._scoring_p
        if self._scoring_p == 'roc_auc':
            self._scorer = make_scorer(roc_auc_score)
        elif self._scoring_p == 'f1':
            self._scorer = make_scorer(f1_score)
        elif self._scoring_p == 'accuracy':
            self._scorer = make_scorer(accuracy_score)
        elif callable(self._scoring_p):
            self._scoring = getattr(self._scoring_p, "name", type(self._scoring_p))
            self._scorer = self._scoring_p
        else:
            self._scoring = self._context._scoring
            self._scorer = self._context._scorer
    
    def fit(self, X_valid, y_valid):
        # setup scorer, after context is set        
        self._setup()
        
        info = pd.DataFrame()
        for c in self._context._clfs:
            sclf = self._context._clfs[c]['sclf']
            clf = sclf.best_estimator_
            if c == 'xgboost':
                clf.fit(self._context._X_training, self._context._y_training)
            info = info.append([{
                'pipeline': self._context._name,
                'scaling': self._context._scaling,
                'scoring': self._context._scoring,
                'name': self._context._clfs[c]['name'],
                'model': clf,
                'params': sclf.best_params_,
                'best_score': sclf.best_score_,                
                'mean_score': sclf.cv_results_['mean_test_score'][sclf.best_index_],
                'std_score': sclf.cv_results_['std_test_score'][sclf.best_index_],
                'run_time': sclf.run_time,
                'validation_scoring': self._scoring,
                'validation_score': self._scorer(clf, X_valid, y_valid)
            }], ignore_index=True)
        self._context._info = info.sort_values(by='validation_score', ascending=False)
        self._context._best = self._context._info.iloc[0].model
        
        if self._context._reporting:
            display(self._context._info)
        
        return self
    
    # concatenate X_training+X_validation, y_traning+y_validation and refit the best model
    def refit(self, X_train_valid, y_train_valid):
        self._context._best.fit(X_train_valid, y_train_valid)
        return self

class ModelTestStep(BinaryClassifierPipelineStep):
    def predict(self, X_test):
        if self._context._best == None:
            raise Exception('You need to fit ModelSelectionStep in order to select the best fitted classifier')
        return self._context._best.predict(X_test)
    
    def score(self, X_test, y_test):
        if self._context._best == None:
            raise Exception('You need to fit ModelSelectionStep in order to select the best fitted classifier')
        
        y_pred = self._context._best.predict(X_test)
        
        info = self._context._info.iloc[0]
        
        return pd.DataFrame().append([{
            'pipeline': info.pipeline,
            'scaling': info.scaling,
            'model_name': info['name'],
            'params': info.params,
            'training_scoring': info.scoring,
            'training_score': info.best_score,
            'validation_scoring': info.validation_scoring,
            'validation_score': info.validation_score,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average="weighted"),
            'roc_auc': roc_auc_score(y_test, y_pred, average="weighted"),
            'afr': afr(y_test, y_pred)
        }])

class BotClassifierPipeline(TransformerMixin, ClassifierMixin):
    """BotClassifierPipeline
    This class defines a fixed pipeline for classifier fit, tuning and selection.
    1. The first step (0 or 'feature_engineering') is where the feature generation, must implement 
       the method `transform(x)` with one position arguments (X). This method must return the transformed
       dataset X with all numeric features.
    2. The second step (1 or `data_splitting`) must implement the method `transform(X, y)`, with two
       positional arguments (X and y). The `transform(X, y)` method must return 6 dataframes: 
       X_train, X_valid, X_test, y_train, y_valid and y_test
    3. The third step (2 or `model_tuning`) must implement the method `fit(X, y)` and perform the parameter
       search for all the models on the training set.
    4. The fourth step (3 or `model_selection`) must implement the method `fit(X, y)`, that will score all
       the models on the validation set and fit the best model with the concatenation of training and 
       validation sets.
    5. The last step (4 or `model_test`) must implement the methods `predict` and `score`, and will score
       the selected model on the test set
    """
    
    def __init__(self, steps, name=None, scaling=None, scoring=None, reporting=False, save_models=False, n_iter=15):
        self._name = name if name != None else str(time.process_time()) 
        self._best = None
        self._info = None
        
        # Define the model saving behaviour for the training and tuning step
        self._save = True if save_models == True else False
        
        # Define the reporting behaviour for the training and tuning step
        self._reporting = True if reporting == True else False
        
        if len(steps) != 5:
            raise Exception('Not all steps are defined')
        
        self._data_split = steps[0][1].set_context(self)
        self._feature_engineering = steps[1][1].set_context(self)
        self._model_tuning = steps[2][1].set_context(self)
        self._model_selection = steps[3][1].set_context(self)
        self._model_test = steps[4][1].set_context(self)
     
    def fit(self, X, y):
        # 1. Data split
        X_training, X_validation, X_test, y_training, y_validation, y_test = self._data_split.transform(X, y)
        
        # 2. Feature engineering
        X_training_t = self._feature_engineering.fit_transform(X_training)
        
        # 3. Model tuning
        self._model_tuning.fit(X_training_t, y_training)
        
        # 4. Model selection
        X_validation_t = self._feature_engineering.transform(X_validation)
        self._model_selection.fit(X_validation_t, y_validation)
        
        # refit selected model with training + validation
        X_train_valid = X_training.append(X_validation, ignore_index=True)
        y_train_valid = y_training.append(y_validation, ignore_index=True)
        X_train_valid_t = self._feature_engineering.fit_transform(X_train_valid)
        self._model_selection.refit(X_train_valid_t, y_train_valid)
        
        # 5. Model test
        self._X_test_t = self._feature_engineering.transform(X_test)
        self._y_test = y_test
        self._score = self._model_test.score(self._X_test_t, y_test)
        
        return self
    
    def score(self):
        if self._best == None:
            raise Exception('You need to run the validation in order to select the best fitted classifier')
        return self._score

    def info(self):
        return self._info.copy()
    
    def predict(self, X_test, y_test):
        if self._best == None:
            raise Exception('You need to run the validation in order to select the best fitted classifier')
        X_test_t = self._feature_engineering.transform(X_test)
        self._score = self._model_test.score(X_test_t, y_test)
        return self

        