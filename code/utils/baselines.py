import pickle
import pandas as pd
import os
from random import random
import numpy as np
from config import config
from utils.majority_vote import majority_vote
from sklearn.model_selection import train_test_split

def create_data_ens_i(agg_train_set, agg_val_set):
    dict_path = os.path.join(config['data_dir'],'dictionary/instance_dict.pickle')
    # print('-->',dict_path)
    with open(dict_path, 'rb') as dict_:
        n_istances = pickle.load(dict_)
    print(n_istances)

    for k in n_istances.keys():
        sampled = agg_train_set.sample(n_istances[k][0]) 
        sampled.to_csv(os.path.join(config['data_dir'], f"ENS-I/ENS-I_{k}_train_set.csv"), index=False)
        sample_rest = agg_train_set[~agg_train_set['id_text'].isin(sampled['id_text'].tolist())]
        to_sample = pd.concat([agg_val_set, sample_rest])
        sampled_ = to_sample.sample(n_istances[k][1])
        sampled_.to_csv(os.path.join(config['data_dir'], f"ENS-I/ENS-I_{k}_validation_set.csv"), index=False)
            
def create_data_ens_a(df):
    dict_path = os.path.join(config['data_dir'],'dictionary/annotator_dict.pickle')
    #print('-->',dict_path)
    with open(dict_path, 'rb') as dict_:
        n_annotator_per_perspective = pickle.load(dict_)
    
    print(n_annotator_per_perspective)
    
    annotators = np.unique(df["id_annotator"])
    
    # For each perspective
    for perspective in n_annotator_per_perspective.keys():
        # Find number of annotators
        n_annotators = n_annotator_per_perspective[perspective]
        # Sample random annotators
        sampled_annotators = np.random.choice(annotators, n_annotators)
        # Select only the annotation of those random annotators
        df_annotators = df[df["id_annotator"].apply(lambda x: True if x in sampled_annotators else False)]
        # Majority vote of these random annotators
        df_annotators_majority = majority_vote(df_annotators, config['parent_text'])
        # Split train and val and save
        df_maj_train, df_maj_val = train_test_split(df_annotators_majority, test_size=0.20, stratify=df_annotators_majority["label"], random_state=10)
        df_maj_train.to_csv(os.path.join(config['data_dir'], f"ENS-A/ENS-A_{perspective}_train_set.csv"), index=False)
        df_maj_val.to_csv(os.path.join(config['data_dir'], f"ENS-A/ENS-A_{perspective}_validation_set.csv"), index=False)
        
        # print(df_maj_train["label"].value_counts())
        # print(df_maj_val["label"].value_counts())