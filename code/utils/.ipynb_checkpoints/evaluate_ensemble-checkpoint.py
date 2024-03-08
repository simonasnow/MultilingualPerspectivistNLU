from sklearn.metrics import classification_report, confusion_matrix
from glob import glob
import csv
from pprint import pprint
import random
import pandas as pd
from config import config
from os.path import basename
import os

def print_metrics(model, gold, pred):
    r = classification_report(
        gold, 
        pred, 
        output_dict=True)
    print (f"{test_set} {seed} {model} {r['0']['precision']:.3f} {r['0']['recall']:.3f} {r['0']['f1-score']:.3f} {r['1']['precision']:.3f} {r['1']['recall']:.3f} {r['1']['f1-score']:.3f} {r['macro avg']['precision']:.3f} {r['macro avg']['recall']:.3f} {r['macro avg']['f1-score']:.3f} {r['accuracy']:.3f}")


def add_results(gold, pred, label, test_set, llm, results, seed):
    r = classification_report(
    gold, 
    pred, 
    output_dict=True)
    results.loc[-1] = [seed, test_set, label, llm, r['0']['precision'], r['0']['recall'], r['0']['f1-score'], r['1']['precision'], r['1']['recall'], r['1']['f1-score'], r['macro avg']['precision'], r['macro avg']['recall'], r['macro avg']['f1-score'], r['accuracy']]
    results.index = results.index + 1 
    results = results.sort_index()


def non_perspectivist(prediction_file):                        
    # aggregated training data model
    pred_gold = []
    with open(prediction_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_gold.append(row["label"])
    return pred_gold


def ensemble_c_high(data):
    # ensemble (max confidence)
    ensemble_max = dict()
    for id, predictions in data.items():
        max_conf = -1.0
        for _, scores in predictions.items():
            if scores["confidence"] > max_conf:
                ensemble_max[id] = scores["label"]
                max_conf = scores["confidence"]
    return ensemble_max

                
def ensemble_c_weight(data):         
    # confidence-based ensemble 
    alpha = 1 # this parameter is not used at the moment
    ensemble = dict()
    for id, predictions in data.items():
        votes = {"0":0, "1":0}
        for _, scores in predictions.items():
            votes[scores["label"]]+=(scores["confidence"]**alpha)
        ensemble[id] = "0" if votes["0"]>votes["1"] else "1"
    return ensemble


def ensemble_m(data):
    # ensemble (majority voting on label)
    ensemble_label = dict()
    for id, predictions in data.items():
        # print(id, predictions)
        votes = {"0":0, "1":0}
        count_0 = 0
        count_1 = 0
        for _, scores in predictions.items():
            if scores['label'] == '0':
                count_0 += 1
            if scores['label'] == '1':
                count_1 += 1
            votes['0']=count_0
            votes['1']=count_1
        ensemble_label[id] = "0" if votes["0"]>votes["1"] else "1"
    return ensemble_label


def compute_results(config, file, predictions_dir, control_name):
    
    if control_name == "ens-a":
        file_name = control_name+'_'+os.path.basename(file).split('_')[0]
        print('----->>>>', file_name)
    elif control_name == "ens-i":
        file_name = control_name+'_'+os.path.basename(file).split('_')[0]
        print('----->>>>', file_name)
    else:
        file_name = os.path.basename(file).split('_')[0]
        print('----->>>>', file_name)

    # dataframe to store all metrics
    results = pd.DataFrame(columns=["seed", "test_set", "model", "llm", "p0", "r0", "f0", "p1", "r1", "f1", "pm", "rm", "fm", "a"])

    for llm in config['llm_list']:
        for seed in config['seed_experiment_list']:

            # read the predictions from perspective-aware models
            data = dict()
            gold_file=''
            for prediction_file in glob(f"{predictions_dir}/*_{seed}_{llm}*.csv"):
                # print('***', prediction_file)
                if os.path.basename(prediction_file).startswith('aggregated'):
                    gold_file = prediction_file
                
                profile = ''
                if control_name == "ens-a" or control_name == "ens-i":
                    profile = os.path.basename(prediction_file).split('_')[1]
                else:
                    profile = os.path.basename(prediction_file).split('_')[0]
                    # exclude predictions from gold models
                    if  profile == 'aggregated':
                        continue
                # print('--> ', profile)
                
                with open(prediction_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # print(row)
                        if not row["id_text"] in data:
                            data[row["id_text"]] = dict()
                            # print(data)
                        data[row["id_text"]][profile] = {"label":row["label"], "confidence":eval(row["confidence"])}
                        
            
            if control_name == "ens-a":
                pred_gold = []
            elif control_name == "ens-i":
                pred_gold = []
            else:
                pred_gold = non_perspectivist(gold_file)

            ensemble_max = ensemble_c_high(data)
            ensemble = ensemble_c_weight(data)
            ensemble_label = ensemble_m(data)

            gold = []
            pred_max = []
            pred_ens = []
            pred_label_max = []

            # print(file)
            with open(file, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        gold.append("0" if row["label"]=="0" else "1")
                        pred_label_max.append(ensemble_label[row['id_text']])
                        pred_max.append(ensemble_max[row['id_text']])
                        pred_ens.append(ensemble[row['id_text']])
                    except:
                        print (row['id_text'])
                        
            if control_name == 'ens-a':
                pred_gold=[]
            elif control_name == 'ens-i':
                pred_gold=[]
            else:
                add_results(gold, pred_gold, "gold", str(file_name), llm, results, seed)
            
            add_results(gold, pred_label_max, "max_label", str(file_name), llm, results, seed)
            add_results(gold, pred_max, "vote", str(file_name), llm, results, seed)
            add_results(gold, pred_ens, "conf", str(file_name), llm, results, seed)
            
            
            # print(results)
            # print(results.columns)
        results.to_csv(os.path.join(os.getcwd(), f'details_results/{llm}_{file_name}_results_ensemble.csv'), sep=',')

    # output the mean over the runs
    print (results.groupby(["test_set", "llm", "model"]).mean())
    # print('--> results saved in ', os.getcwd())

    df = results.groupby(["test_set", "llm", "model"]).mean()
    df.to_csv(os.path.join(os.getcwd(), f'results/{file_name}_results_ensemble.csv'), sep=',')