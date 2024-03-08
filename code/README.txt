To run experiments, run the script run.py.

The commands are:
1. datasets.create_dataset(config) = starting from a raw non-aggregated datasets, this function creates all the datasets: standard (with majority voting aggregated labels), and perspectives-based (the dimensions for the profiles should be explicited in the config.py file)
2. training_models.get_predictions(config) = on the basis of the created datasets, this command performs all the experiments for each setting (in-domain, out-of-domain, ens-i and ens-a), obtaining the predictions in a specific folder ('./predictions/')
3. performance.get_results(config) = create the ensemble and a file containing the average of all the results for each setting.