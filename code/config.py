hyperparameters = {
    "lr": {
        "bert-base-uncased": 5e-05,
        "distilbert-base-uncased": 5e-06,
        "roberta-base": 5e-06,
    },
    "epochs": 20,
    "batch_size": 16,
    "batch_size_eval": 64,
    "patience": 2,
}

config = {
    "corpus_file": "./corpus/EPICorpus.csv", #raw disaggregated dataset
    "traits": ["Generation", "Gender", "Nationality"], #dimensions to characterize the annotators
    "I-ENS_baseline": False, #baseline based on randomizing instances
    "A-ENS_baseline": False, #baseline based on randomizing annotators
    "parent_text": True, #for short conversations
    "save_model":False,
    "data_dir": "./EPIC/split_data/", #direction to save all the datasets (standard and perspective-based)
    "prediction_dir_ID": "./predictions/in_domain/",#default
    "prediction_dir_OOD": "./predictions/out_of_domain/",#if out_of_domain is not a empty list
    "models_dir": "./models/", #default
    "dictstate_dir": "./dictstates/", #default
    "out_of_domain": ['./other_corpora/irony/SemEval_test_set.csv'], # for cross-dataset predictions: the filename of test set should finish in "_test_set.csv" and should contain the following columns: [id_text, text, label]
    "seed_data" : 42, #to reproduce the split of train/val/test sets
    "seed_experiment_list" : [10,11,12,13,14,15,16,17,18,19],
    'llm_list': ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base'],
}
