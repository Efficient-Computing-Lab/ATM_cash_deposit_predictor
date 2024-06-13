#Part of ATMO-MoRe ATM Load Predictor, a system that models ATM load and their need for resupply.
#Copyright (C) 2024  Evangelos Psomakelis
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU Affero General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU Affero General Public License for more details.
#
#You should have received a copy of the GNU Affero General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
from atm_load_prediction.data_handler import DataHandler, Preprocessor
from atm_load_prediction.utils import csv_to_supervised, printProgressBar
from atm_load_prediction.load_predictor import ATMOPredictor
from atm_load_prediction.models import LinearModel
from atm_load_prediction.evaluator import evaluate_models_timelag, evaluate_lstm
import pandas
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("command", choices=[
    'import_data', 
    'evaluate_classic_models', 
    'evaluate_deep_models', 
    'apply_models'
])
parser.add_argument("--data",nargs='?')
args=parser.parse_args()

if args.command == "import_data":

    try:
        dh = DataHandler()
        dh.atm_renew_data()
        print("Data loaded.")
    except Exception as ex:
        print(f"Error on data loading: {ex}")

elif args.command == "evaluate_classic_models":

    resses = []
    evaluation_datasets = Preprocessor.initiate_evaluation_datasets()
    Path(f"results").mkdir(parents=True, exist_ok=True)  
    for i in range(len(LinearModel.algorithms)):
        model = ''
        model = LinearModel.algorithms[i][0].__name__
        try:
            print(f'----------------  RUN {i}: {model} -----------------')
            res = evaluate_models_timelag(algorithm=i,**evaluation_datasets)
            res.to_csv(f"./results/evaluation_results_class_{i}.csv",sep=";",decimal=",",encoding="utf8")
            res['algorithm'] = model
            resses.append(res)
            print({'mean_accuracy':res['mean_accuracy'].mean(),'r2':res['r2'].mean(),'mean_absolute_error':res['mean_absolute_error'].mean()})
        except Exception as ex:
            print(f"Model evaluation error: {ex}")
            continue
    pandas.concat(resses).to_csv(f"./results/evaluation_results_full.csv",sep=";",decimal=",",encoding="utf8")

elif args.command == "evaluate_deep_models":

    timeseries = Preprocessor.initiate_evaluation_datasets()
    Path(f"results").mkdir(parents=True, exist_ok=True)  
    for i in range(1,7):
        results = evaluate_lstm(timeseries,i)
        res = pandas.DataFrame.from_records(results)
        res.to_csv(f"./results/evaluation_results_lstm_{i}.csv",sep=";",decimal=",",encoding="utf8")

elif args.command == "apply_models":

    current_state_data_path = args.data
    current_state_dict = None
    if current_state_data_path is None:
        print("Please specify a filepath for the current state data csv.")
        exit(0)
    else:
        try:
            current_state_dict = csv_to_supervised(
                current_state_data_path,
                converter=Preprocessor.test_to_supervised,
                train_features=['day_of_week','day_of_month','month','workday','holiday','value']
            )
        except Exception as ex:
            print(f"Error on current state data parsing: {ex}")
            exit(0)

    predictor = ATMOPredictor(autotrain=True)
    
    common_atms = [atm for atm in current_state_dict if atm in predictor.atm_codes]
    missing_models = [atm for atm in current_state_dict if atm not in common_atms]
    if len(missing_models) > 0:
        print(f"No training data for ATMs: {missing_models}")
    atm_index = 0
    print(f"Running predictor for ATMs: {common_atms}")
    results = []
    print("completion...")
    for atm in common_atms:
        printProgressBar(atm_index,len(common_atms),suffix=atm)
        current_state = current_state_dict[atm]
        if predictor.is_atm_due(atm_code=atm,current_state=current_state):
            results.append(atm)
        atm_index += 1
    printProgressBar(atm_index,len(common_atms),suffix=atm)
    print("Completed!")
    print()
    print(f"ATMs due for supply: {results}")