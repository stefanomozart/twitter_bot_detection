import os
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import FunctionTransformer

def read_twitter_dataset(path='.', json_file=None, tsv_file=None):
    if json_file != None:
        if tsv_file == None:
            raise Exception("You need to inform either a folder `path`, or both values for `json_file` and `tsv_file`")
        else:
            json_path = json_file
            tsv_path = tsv_file
    elif tsv_file != None:
        raise Exception("You need to inform either a folder `path`, or both values for `json_file` and `tsv_file`")
    else:
        for (root, _, files) in os.walk(path):
            # Variables to store json and tsv data
            _json, tsv = None, None

            if files is not None and len(files) == 2:
                tsv_path = files[0] if str(files[0]).endswith("tsv") else files[1]
                json_path = files[0] if str(files[0]).endswith("json") else files[1]
            else:
                raise FileExistsError("The provided path does not contain a valid Twitter dataset")

            tsv_path, json_path = os.path.join(root, tsv_path), os.path.join(root, json_path)

    tsv = pd.read_csv(tsv_path, sep="\t", names=["id", "label"]).set_index("id")

    with open(json_path, 'r') as f:
        _json = json.load(f)
        _json = pd.json_normalize(_json)
        
    def is_bot(x, tsv):
        idx = pd.Index([x["user.id"]])
        if not idx.isin(tsv.index):
            return np.NaN

        return tsv.loc[idx]["label"].values[0] == "bot"

    # Create the label column
    _json["is_bot"] = _json.apply(lambda x: is_bot(x, tsv), axis=1)

    return _json

def twitter_feature_generation(df_):
    df = df_.copy()
    count_digits = lambda s: sum(item.isdigit() for item in s)
    
    df["user.screen_name_length"] = df["user.screen_name"].str.len()
    df["user.screen_name_number_digits"] = df["user.screen_name"].apply(count_digits)
    df["user.name_length"] = df["user.name"].str.len()
    df["user.name_number_digits"] = df["user.name"].apply(count_digits)
    df["user.description_length"] = df["user.description"].str.len()

    return df

twitter_feature_generation_transformer = FunctionTransformer(twitter_feature_generation)

def twitter_feature_selection(df):
    return df[[
        "user.statuses_count",
        "user.followers_count",
        "user.friends_count",
        "user.favourites_count",
        "user.listed_count",
        "user.default_profile",
        "user.profile_use_background_image",
        "user.verified",
        "user.screen_name_length",
        "user.screen_name_number_digits",
        "user.name_length",
        "user.name_number_digits",
        "user.description_length"
    ]].copy()

twitter_feature_selection_transformer = FunctionTransformer(twitter_feature_selection)
