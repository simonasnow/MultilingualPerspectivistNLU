import pandas as pd
import os

def generate_df(df, list_users, list_labels, method):
    user_label_dict = dict(zip(list_users, list(list_labels)))
    df_cluster = pd.DataFrame(user_label_dict.items(), columns=['id_annotator', 'cluster'])
    df_method = df.merge(df_cluster[['id_annotator', 'cluster']], on='id_annotator', how='left')

    # df_method_ = df_method[["user", "label", "id_original", "text", "parent_id_original", "parent_text","Generation", "Sex", "Nationality", "cluster"]]
    # df_method_ = df_method_.rename(columns={"user": "id_annotator", "id_original":"id_text", "parent_id_original": "id_parent", "parent_text":"parent", "Sex":"Gender"})
    return df_method