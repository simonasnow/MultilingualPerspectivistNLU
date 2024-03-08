#!/usr/bin/env python

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaConfig, BertConfig, DistilBertConfig, TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
import pandas as pd
import numpy as np
from datasets import concatenate_datasets
from utils.Data import generate_data
from utils.Globals import device
from glob import glob
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils import compute_class_weight
import torch
from config import hyperparameters, config
import csv

config_map = {
    "roberta-base": RobertaConfig,
    "distilbert-base-uncased": DistilBertConfig,
    "bert-base-uncased": BertConfig
}

class CustomTrainer(Trainer):
    """ Custom Trainer class to implement a custom looss function
    """
    def set_class_weights(self, class_weights):
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Focal Loss: https://arxiv.org/abs/1708.02002
        """
        gamma = 5.0
        alpha = .2
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights)).to(device)
        BCEloss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        # Focal Loss
        pt = torch.exp(-BCEloss) # prevents nans when probability 
        loss = alpha * (1-pt)**gamma * BCEloss
        return (loss, outputs) if return_outputs else loss
                
def compute_metrics(p):    
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1} 

def get_model(llm):    
    config_model = config_map[llm].from_pretrained(llm)
    return AutoModelForSequenceClassification.from_pretrained(
        llm, config=config_model)

def train(data, llm, profile, seed, save):
    set_seed(seed)

    # compute class weights
    try:
        class_weights = compute_class_weight(
            "balanced", 
            classes=[0, 1], 
            y=data["validation"]["labels"].float().numpy()).astype("float32")
    except:
        class_weights = np.array([1,1]).astype("float32")
        
    # build model
    model = get_model(llm)
        
    # training arguments
    training_args = TrainingArguments(
            output_dir = config['models_dir'], #it is required by TrainingArguments
            num_train_epochs=hyperparameters["epochs"],  
            learning_rate = hyperparameters["lr"][llm],
            per_device_train_batch_size=hyperparameters["batch_size"],  
            per_device_eval_batch_size=hyperparameters["batch_size_eval"], 
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            logging_strategy="epoch",
            load_best_model_at_end = True,
            seed = seed,
            )
    
    trainer = CustomTrainer(
            model=model,                         
            args=training_args,                  
            train_dataset=data["train"],         
            eval_dataset=data["validation"],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=hyperparameters["patience"])],
        )
    
    trainer.set_class_weights(class_weights)
    trainer.train()
    
    if save:
        path_model = os.path.join(config['models_dir'], f"{profile}_{seed}_{llm}/")
        trainer.save_model(path_model)
        print('--> saving the best model')
        
    #trainer.evaluate()
    return trainer

def softmax(vec):
    exponential = np.exp(vec)
    probabilities = exponential / np.sum(exponential)
    return probabilities
  
def predict(trainer, profile, test_set, data_dir, seed, llm, control_name):
    test_data = generate_data(test_set, "test", data_dir, llm, ["0", "1"], config["parent_text"])
    predictions = trainer.predict(test_data)
    
    path_output_main = "./predictions/" 
    predictions_dir = path_output_main
    
    if control_name.startswith('aggregated'):
        if test_set.startswith('aggregated'):
            predictions_dir = config['prediction_dir_ID']
        
    elif control_name.startswith('ENS-I'): 
        if test_set.startswith('aggregated'):
            predictions_dir = os.path.join(path_output_main, "ENS_I-in_domain")
        else:
            predictions_dir = os.path.join(path_output_main, "ENS_I-out_of_domain")
    
    elif control_name.startswith('ENS-A'):
        if test_set.startswith('aggregated'):
            predictions_dir = os.path.join(path_output_main, "ENS_A-in_domain")
        else:
            predictions_dir = os.path.join(path_output_main, "ENS_A-out_of_domain")
    else:
        predictions_dir = config['prediction_dir_OOD']
    
    
    with open(f"{predictions_dir}/{profile}_predictions_{seed}_{llm}_{test_set}.csv", "w") as fo:
        writer = csv.DictWriter(
            fo, 
            fieldnames = [
                "id_text", 
                "label", 
                "confidence"])
        writer.writeheader()

        for i, instance in enumerate(test_data):
            p0,p1 = softmax(predictions.predictions[i])
            pred = np.argmax(predictions.predictions[i])
            diff = (p1-p0)   if pred==1 else (p0-p1)
            conf = ((p1-p0)/abs(p1+p0))  if pred==1 else ((p0-p1)/abs(p0+p1))

            writer.writerow({
                "id_text": instance['id_text'], 
                "label": pred,
                "confidence": conf                
            })

def training_setting(path_input, config):
    seed_list = config['seed_experiment_list']
    llm_list = config['llm_list']
    
    print('--> directory in analysis: ', path_input)
    for training_file in glob(os.path.join(path_input, "*_train_set.csv")):
        
        save_model = False
        ens_a = False
        ens_i = False
        train_set = ''
        file_name = os.path.basename(training_file).split("_")
        
        if "ENS-A" in file_name:
            train_set = file_name[0]+'_'+file_name[1]
            save_model = False
            ens_a = True
        elif "ENS-I" in file_name:
            train_set = file_name[0]+'_'+file_name[1]
            save_model = False
            ens_i = True
        else:
            train_set = file_name[0]
            save_model = config['save_model']
            ens_a = False
            ens_i = False
 
        print (f"*** training set: {train_set} ***")

        if train_set != 'Unknown': #in EPIC dataset                
            for llm in llm_list:
                # read data
                print('--> data generation on ', llm)
                data = {split: generate_data(train_set, split,  path_input, llm, ["0", "1"], config["parent_text"]) for split in ["train", "validation"]}

                for seed in seed_list:
                    print(f'--> llm: {llm} in a seed of: {seed}')
                    prediction_dir = config["prediction_dir_ID"]

                    if os.path.isfile(f"{prediction_dir}/{train_set}_predictions_{seed}_{llm}_aggregated_test_set.csv"):
                        print ("already done")
                    else:
                        trainer = train(data, llm, train_set, seed, save_model)

                        test_set_file = [i for i in glob(os.path.join(config["data_dir"], "aggregated_test_set.csv"))][0] 
                        test_set_name = os.path.basename(test_set_file).split("_")[0]
                        if ens_a:
                            control_name = 'ENS-A_'+test_set_name
                        elif ens_i:
                            control_name = 'ENS-I_'+test_set_name
                        else:
                            control_name = test_set_name
                        print (f"*** in-domain test set: {test_set_name} ***")
                        predict(trainer, train_set, test_set_name, config["data_dir"], seed, llm, control_name)

                        if len(config["out_of_domain"]) >= 1:
                            for test_set_ood in config['out_of_domain']:
                                test_set_name = os.path.basename(test_set_ood).split("_")[0]
                                data_dir = os.path.dirname(test_set_ood)
                                print (f"*** out-of-domain test set: {test_set_name} ***")
                                if ens_a:
                                    control_name = 'ENS-A_'+test_set_name
                                elif ens_i:
                                    control_name = 'ENS-I_'+test_set_name
                                else:
                                    control_name = test_set_name
                                predict(trainer, train_set, test_set_name, data_dir, seed, llm, control_name)
                        del(trainer)
    
    print('training done')
                      
def inference(model, file_test_set, profile, test_set, seed, llm, prediction_dir):
    test_data = pd.read_csv(file_test_set, sep=',', header=0)

    with open(f"{prediction_dir}/{profile}_predictions_{seed}_{llm}_{test_set}.csv", "w") as fo:
        writer = csv.DictWriter(
            fo, 
            fieldnames = [
                "id_text", 
                "label", 
                "confidence"])
        writer.writeheader()
        
        for i, t in enumerate(testing_set['text'].tolist()): #da implementare il parent_text
            tokenized = tokenize_inference(t, llm, config['parent_text'])
            result = model(tokenized['input_ids'],tokenized['attention_mask'])
            p0,p1 = softmax(result['logits'].detach()).item()
            pred = torch.argmax(result['logits'].detach()).item()
            diff = (p1-p0)   if pred==1 else (p0-p1)
            conf = ((p1-p0)/abs(p1+p0))  if pred==1 else ((p0-p1)/abs(p0+p1))

            writer.writerow({
                "id_text": testing_set['id_text'].tolist()[i], 
                "label": pred,
                "confidence": conf                
            })

            
def load_model(dir_models, file_test_set, prediction_dir, file_name):
    for file in glob(dir_models):
        name_model = os.path.dirname(file).split('_')
        profile= name_model[0]
        seed = name_model[1]
        llm = name_model[2]
        
        model = get_model(llm)
        model.load(file)
        model.eval()
        inference(model, file_test_set, profile, file_name, seed, llm, prediction_dir)

    