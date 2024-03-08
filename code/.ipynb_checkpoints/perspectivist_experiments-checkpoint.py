import pandas as pd
from utils.model import training_setting, predict
from utils.evaluate_ensemble import non_perspectivist, ensemble_m, ensemble_c_high, ensemble_c_weight, add_results, compute_results
import csv
from config import config
from glob import glob
import os
from utils.split_data import split_datasets
from utils.baselines import create_data_ens_i, create_data_ens_a
import warnings
warnings.filterwarnings('ignore')

def check_dir(path):
    if os.path.exists(path):
        print(path)
    else:
        print('--> creating directions...')
        os.mkdir(path)

class datasets:
    '''
    input: dataset directory with file in csv containing the following columns:
    id_annotator, label, id_text, text, id_parent, parent_text, trait_1, trait_2, trait_3
    
    return: an aggregated dataset (through majority voting) and a dataset for each trait (already split in training, validation and test set)
    
    example:
    - trait: generation
    - profiles: boomers, generation X, Y, Z
    
    Here you can also create data for the baselines I-ENS e A-ENS.

    '''

    def create_dataset(config):
        path_input = config["corpus_file"]
        path_output = config["data_dir"]
        
        check_dir(path_output)
        
        traits = config['traits']
        parent = config['parent_text']
        seed = config['seed_data']
        
        df = pd.read_csv(path_input, sep=',', header=0)
        print(df.shape)
        if df['label'].tolist()[1] != '1' or df['label'].tolist()[1] != '0':    
            print('Remember: Label columns should be string values of 1 and 0')
        
        df_selected, agg_train_set, agg_val_set = split_datasets (df, seed, path_output, traits, parent)

        if config["I-ENS_baseline"]:
            check_dir(os.path.join(path_output, "ENS-I/"))
            create_data_ens_i(agg_train_set, agg_val_set)
            
        if config["A-ENS_baseline"]:
            check_dir(os.path.join(path_output, "ENS-A/"))
            create_data_ens_a(df_selected)
        
        print(path_output)
    
class training_models:
    
    '''
    input: directory with all the datasets
    create: all the models
    return: all predictions for each model
    '''
    
    def get_predictions(config):
        
        path_output_ID = config['prediction_dir_ID']
        path_output_OOD = config['prediction_dir_OOD']
        path_models = config['models_dir']
        
        #creazione of directories if doesn't exist
        #these are used inside model.py
        check_dir(path_models)
        path_output_main = "./predictions/" 
        check_dir(path_output_main)
        check_dir(path_output_ID)
        if len(config["out_of_domain"]) >= 1:
            print('--> out-of-domain setting took into account')
            check_dir(path_output_OOD)
        
        #in-domain e out-of-domain setting:
        path_input = config["data_dir"]
        training_setting(path_input, config) 
        
        if config["I-ENS_baseline"]:
            check_dir(os.path.join(path_output_main, "ENS_I-in_domain"))
            check_dir(os.path.join(path_output_main, "ENS_I-out_of_domain"))
            training_setting(os.path.join(config["data_dir"], "ENS-I/"), config)

        if config["A-ENS_baseline"]:
            check_dir(os.path.join(path_output_main, "ENS_A-in_domain"))
            check_dir(os.path.join(path_output_main, "ENS_A-out_of_domain"))
            training_setting(os.path.join(config["data_dir"], "ENS-A/"), config)

                            
class performance:
    '''
    input: the predictions of all the models
    output: results also of the ensembles
    '''
    
    def get_results(config):
        path_output_main = "./predictions/"
       
        names_dir = [name for name in os.listdir(path_output_main) if os.path.isdir(path_output_main+name)]
        # print(names_dir)
        for predictions_dir in names_dir :
                print('-->', predictions_dir)
                
                if predictions_dir == 'ENS_A-in_domain':
                    control_name = 'ens-a'
                    file = os.path.join(config['data_dir'], "aggregated_test_set.csv")
                    print('looking at ', file)
                    compute_results(config, file, os.path.join(path_output_main, predictions_dir), control_name)

                elif predictions_dir == 'ENS_A-out_of_domain':
                    control_name = 'ens-a'
                    for file in config['out_of_domain']:
                        print('looking at ', file)
                        compute_results(config, file,os.path.join(path_output_main, predictions_dir), control_name)
            
                elif predictions_dir == 'ENS_I-in_domain':
                    control_name = 'ens-i'
                    file = os.path.join(config['data_dir'], "aggregated_test_set.csv")
                    print('looking at ', file)                    
                    compute_results(config, file, os.path.join(path_output_main, predictions_dir), control_name)

                elif predictions_dir == 'ENS_I-out_of_domain':
                    control_name = 'ens-i'
                    for file in config['out_of_domain']:
                        print('looking at ', file)
                        compute_results(config, file, os.path.join(path_output_main, predictions_dir), control_name)
            
                elif predictions_dir=='in_domain':
                    control_name = ''
                    file = os.path.join(config['data_dir'], "aggregated_test_set.csv")
                    print('looking at ', file)
                    compute_results(config, file, os.path.join(path_output_main, predictions_dir), control_name)

                else:
                    control_name = ''
                    predictions_dir = config['prediction_dir_OOD']
                    for file in config['out_of_domain']:
                        print('looking at ', file)
                        compute_results(config, file, predictions_dir, control_name)
            
                