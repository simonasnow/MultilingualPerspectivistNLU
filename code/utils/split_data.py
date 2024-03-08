import pandas as pd
import os
import pickle
from utils.majority_vote import majority_vote

def split (df, seed):
    df['label'] = df['label'].astype(str)
    l_pos = df.loc[df['label'] == '1']
    l_neg = df.loc[df['label'] == '0']
    # print(l_pos)
    # print(l_neg)

    value_pos = round((l_pos.shape[0] * 80) / 100)  # 70/80
    value_neg = round((l_neg.shape[0] * 80) / 100)
    training = pd.concat([l_pos[:value_pos], l_neg[:value_neg]])
    # print(value_pos, value_neg)

    l_pos_1 = training.loc[training['label'] == '1']
    l_neg_1 = training.loc[training['label'] == '0']
    value_pos_1 = round((l_pos_1.shape[0] * 80) / 100)  # 80/70
    value_neg_1 = round((l_neg_1.shape[0] * 80) / 100)
    # print(value_pos_1, value_neg_1)

    train = pd.concat([l_pos_1[:value_pos_1], l_neg_1[:value_neg_1]])
    val = pd.concat([l_pos_1[value_pos_1:], l_neg_1[value_neg_1:]])
    test = pd.concat([l_pos[value_pos:], l_neg[value_neg:]])
    train_ = train.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_ = val.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_ = test.sample(frac=1, random_state=seed).reset_index(drop=True)
    # print(train,'\n', val,'\n', test)

    return train_, val_, test_

def split_datasets (df, seed, DIR, traits, parent_text):
    
    df_gold = majority_vote (df, parent_text)
    df_gold_ = df_gold.loc[df_gold['label'] != 'both']
    df_gold_ = df_gold_.reset_index(drop=True)
    # print(df_gold)
    # print(df_gold_)

    df_gold_shuffled = df_gold_.sample(frac=1, random_state=seed).reset_index(drop=True)
    agg_train_set, agg_val_set, agg_test_set = split(df_gold_shuffled, seed)
    agg_train_set.to_csv (os.path.join(DIR, f"aggregated_train_set.csv"), index=False)
    agg_val_set.to_csv (os.path.join(DIR, f"aggregated_validation_set.csv"), index=False)
    agg_test_set.to_csv (os.path.join(DIR, f"aggregated_test_set.csv"), index=False)
        
    df_selected = df[~df['id_text'].isin(agg_test_set['id_text'].tolist())]
    
    print(df_selected.shape)

    print('texts from the aggregated test set in the df_selected:')
    for i in agg_test_set['id_text'].tolist():
        if i in df_selected['id_text'].tolist():
            print(i)

    path = os.path.join(DIR, f"dictionary")
    if not os.path.exists(path):
        os.mkdir(path)

    annotator_dict = {}
    instance_dict={}
    for trait in traits:
        print('-- ', trait)
        profile_count = df[[trait, 'id_annotator']].drop_duplicates()
        profile_dict = profile_count[trait].value_counts().to_dict()
        annotator_dict.update(profile_dict)
        with open(os.path.join(path,"annotator_dict.pickle"), "wb") as handle:
            pickle.dump(annotator_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path,"annotator_dict.pickle"), "rb") as handle:
            deserialized_dict_1 = pickle.load(handle)

        for profile in df[trait].unique():
            print('---- ', profile)
            df_ = majority_vote(df_selected.loc[df_selected[trait] == profile], parent_text)
            # print('mj', df_.shape)
            # print(df_)
            df_1 = df_.loc[df_['label'] != 'both']
            # print(df_1.shape)
            if df_1.shape[0] == 0:
                print('Attention!', profile, 'dataset : ', df_1.shape, 'Check labels values in your original dataset. The label column should contain string values of 0 and 1.')
            df_1 = df_1.reset_index(drop=True)
            # print(df_1.shape)

            p_train_set, p_validation_set, p_test_set = split(df_1, seed)
            p_train_set_complete = pd.concat([p_train_set, p_test_set])
            
            p_train_set_complete.to_csv(os.path.join(DIR,f"{profile}_train_set.csv"), index=False)
            p_validation_set.to_csv(os.path.join(DIR,f"{profile}_validation_set.csv"), index=False)
            # p_test_set.to_csv(os.path.join(DIR,f"{profile}_test_set.csv"), index=False) #test set based on profile not needed

            key = profile
            value_train = len(p_train_set)+len(p_test_set) #da ricreare i dizionari di MHS
            value_validation = len(p_validation_set)
            instance_dict[key] = value_train, value_validation
            with open(os.path.join(path,"instance_dict.pickle"), "wb") as handle:
                pickle.dump(instance_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path,"instance_dict.pickle"), "rb") as handle:
                deserialized_dict_2 = pickle.load(handle)

    # print("ANNOTATOR DICTIONARY -->", annotator_dict)
    # print("INSTANCE DICTIONARY -->", instance_dict)

    print('data have been saved...')
    # print(DIR)
    return df_selected, agg_train_set, agg_val_set 