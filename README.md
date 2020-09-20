# Machine vs bot: tuning Machine Learning models to detect bots on Twitter
This repository contains the code for the pipeline and the experiments presented in the paper 
"Machine vs Bot: tuning machine learning models to detect bots on Twitter" submited to the 5th 
Workshop on Computer Networks and Communication Systems (WCNPS) 2020.

Authors:

- Stefano M P C Souza - Department of Electrical Engineering, University of Brasília (UnB), 
  Brasília, Brazil
- Tito Resende - Institute of Computing, University of Campinas (IC/Unicamp), Campinas, Brazil
- José Nascimento - IC/Unicamp, Campinas, Brazil
- Levy G Chaves - IC/Unicamp, Campinas, Brazil
- Darlinne H P Soto - IC/Unicamp, Campinas, Brazil
- Soroor Salavati - IC/Unicamp, Campinas, Brazil


    Abstract — Bot generated content on social media canspread fake news and hate speech, 
    manipulate public opinion and influence the community on relevant topics, such as 
    elections. Thus, bot detection in social media platforms plays an important role for the 
    health of the platforms and for the well-being of societies. In this work, we approach the 
    detection of bots on Twitter as a binary output problem through the analysis of account 
    features. We propose a pipeline for feature engineering and model training, tuning and 
    selection. We test ourpipeline using 3 publicly available bot datasets, comparing the 
    performance of all trained models with the model selected at the end of our pipeline.

    Keywords — machine learning, hyper-parameter tuning, random search, bot detection

The `data` folder contains three datasets copied from previous works:

1. **cresci-stock**: S. Cresci, F. Lillo, D. Regoli, S. Tardelli, and M. Tesconi, “Cash-tag 
piggybacking:  Uncovering spam and bot activity in stockmicroblogs on twitter", ACM 
Transactions on the Web (TWEB)", vol. 13, no. 2, pp. 1–27, 2019.
2. **botwiki**: K.-C. Yang, O. Varol, P.-M. Hui, and F. Menczer, “Scalableand generalizable 
social bot detection through data selection”, Proceedings of the AAAI Conference on Artificial
Intelligence, vol. 34, no. 01, p. 1096–1103, Apr 2020.
4. **cresci-rtbust**: M. Mazza, S. Cresci, M. Avvenuti, W. Quattrociocchi, and M. Tesconi, 
“Rtbust: Exploiting temporal patterns for botnet detection on twitter", in Proceedings of the 
10th ACM Conference on Web Science, 2019, pp. 183–192.

## Reading the account information received from the Twitter API

We provide the utility function `read_twitter_dataset`, to read the json format from the 
Twitter API into a pandas dataframe. The datasets must contain a `json` file with accounts in 
the format provided by Twitter API. Each dataset must also contain a `tsv` or `csv` file with 
two columns: the account ID and a boolean with the value True indicating that the account was 
identified as a bot. To use it, simply load from the `twitter_utils.py` file.

```{python}
from twitter_utils import read_twitter_dataset

df = read_twitter_dataset('/path/to/dataset/folder')
```

## Loading the pipeline and its steps

We have placed the code for the pipeline in the file `machine_vs_bot.py`. We organized the code 
in a way that we could clearly identify the pipeline steps presented on the paper. For those 
familiar with scikit-learn's Pipeline, the code will be realy easy to read and understand.

```{python}
from machine_vs_bot import DataSplitStep, FeatureEngineeringStep
from machine_vs_bot import ModelTuningStep, ModelSelectionStep, ModelTestStep
from machine_vs_bot import BotClassifierPipeline

pipeline = BotClassifierPipeline([
    ('data_split', DataSplitStep()),
    ('feature_engineering', FeatureEngineeringStep()),
    ('model_tuning', ModelTuningStep()),
    ('model_selection', ModelSelectionStep()),
    ('model_test', ModelTestStep())
])

y_pred = pipeline.fit(X, y).predict(X_test)
```


## Feature engineering specific to Twitter bot accounts

We have also created transformers customized for twitter account feature generation and 
selection. For the sake of readability, we have placed the code specific to Twitter on the 
`twitter_utils.py` file. The pipeline can be easily extended to fit bot detection classifiers
for other platforms just by creating specific transformer for each platform.

```{python}
# Feature generation: create new columns on the dataframe
from twitter_utils import twitter_feature_generation_transformer

# Feature selection: select the columns that will be fed to the
# classifier for training
from twitter_utils import twitter_feature_selection_transformer

# Feature scaling: any object implementing sklearn.preprocessing.TransformerMixin
from sklearn.preprocessing import StandardScaler

pipeline = BotClassifierPipeline([
    ('data_split', DataSplitStep()),
    ('feature_engineering', FeatureEngineeringStep(
            feature_generation=twitter_feature_generation_transformer,
            feature_selection=twitter_feature_selection_transformer,
            feature_scaling=StandardScaler()
        )),
    ('model_tuning', ModelTuningStep()),
    ('model_selection', ModelSelectionStep()),
    ('model_test', ModelTestStep())
])

y_pred = pipeline.fit(X, y).predict(X_test)
```
## Experiments

In the `machine_vs_bot.ipynb` notebook, you will find the experiments discussed on the paper. 
We run our experiments with 3 different values for feature scaling: 'none', 'standard' (for 
`StandarScaler` normalization trasnform) and 'min-max' (for the `MinMaxScaler`); and 3 
different cross-validation scoring functions: `roc_auc_score`, `f1_score` and `accuracy_score`. 
We fixed `roc_auc_score` as the only validation scoring function in order to be able to compare 
all the models.

