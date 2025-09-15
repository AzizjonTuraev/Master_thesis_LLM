import pandas as pd
import numpy as np
import os

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder


def mle_adult(
        org_test_df : pd.DataFrame,
        org_train_df : pd.DataFrame,
        ml_model,
        path_to_synthetic_data : str,       
):
    
    """
    ml_model : should be provided like that - CatBoostClassifier()
    """
    
    le_workclass = LabelEncoder()
    le_marital_status = LabelEncoder()
    le_occupation = LabelEncoder()
    le_race = LabelEncoder()
    le_gender = LabelEncoder()
    le_native_country = LabelEncoder()
    le_income = LabelEncoder()

    column_name = "workclass"
    org_train_df[column_name] = le_workclass.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_workclass.transform(org_test_df[column_name])

    column_name = "marital-status"
    org_train_df[column_name] = le_marital_status.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_marital_status.transform(org_test_df[column_name])

    column_name = "occupation"
    org_train_df[column_name] = le_occupation.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_occupation.transform(org_test_df[column_name])

    column_name = "race"
    org_train_df[column_name] = le_race.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_race.transform(org_test_df[column_name])

    column_name = "gender"
    org_train_df[column_name] = le_gender.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_gender.transform(org_test_df[column_name])

    column_name = "native-country"
    org_train_df[column_name] = le_native_country.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_native_country.transform(org_test_df[column_name])

    column_name = "income"
    org_train_df[column_name] = le_income.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_income.transform(org_test_df[column_name])

    X_train = org_train_df.drop("income", axis=1)
    Y_train = org_train_df["income"]

    X_test_org = org_test_df.drop("income", axis=1)
    Y_test_org = org_test_df["income"]

    ml = ml_model
    ml.fit(X_train, Y_train)
    y_predict_test_org= ml.predict(X_test_org)
    models_name = os.listdir(path_to_synthetic_data)



    outputs = []
    for model in models_name:

        model_output = {"model_name" : model}
        # print(model)

        for i in range(1, 6):
            # print(i)
            synt_test_df = pd.read_csv(f"{path_to_synthetic_data}/{model}/samples/test/sample{i}.csv", sep=",")[:len(X_test_org)]

            if "Unnamed: 0" in synt_test_df.columns:
                synt_test_df = synt_test_df.drop('Unnamed: 0', axis=1)

            column_name = "workclass"
            synt_test_df[column_name] = le_workclass.transform(synt_test_df[column_name])
            column_name = "marital-status"
            synt_test_df[column_name] = le_marital_status.transform(synt_test_df[column_name])
            column_name = "occupation"
            synt_test_df[column_name] = le_occupation.transform(synt_test_df[column_name])
            column_name = "race"
            synt_test_df[column_name] = le_race.transform(synt_test_df[column_name])
            column_name = "gender"
            synt_test_df[column_name] = le_gender.transform(synt_test_df[column_name])
            column_name = "native-country"
            synt_test_df[column_name] = le_native_country.transform(synt_test_df[column_name])
            column_name = "income"
            synt_test_df[column_name] = le_income.transform(synt_test_df[column_name])

            X_test_synt = synt_test_df.drop("income", axis=1)
            Y_test_synt = synt_test_df["income"]

            y_predict_syn_test = ml.predict(X_test_synt)
            f1_score_micro = f1_score(Y_test_synt, y_predict_syn_test, average='micro')
            f1_score_macro = f1_score(Y_test_synt, y_predict_syn_test, average='macro')
            f1_score_weighted = f1_score(Y_test_synt, y_predict_syn_test, average='weighted')
            accuracy_ = accuracy_score(Y_test_synt, y_predict_syn_test)
            precision_ = precision_score(Y_test_synt, y_predict_syn_test)
            recall_ = recall_score(Y_test_synt, y_predict_syn_test)

            model_output[i] = {
                "f1_score_micro": f1_score_micro,
                "f1_score_macro": f1_score_macro,
                "f1_score_weighted": f1_score_weighted,
                "accuracy": accuracy_,
                "precision": precision_,
                "recall": recall_,
            }

        outputs.append(model_output)




    stats = {}

    for model in outputs:
        f1_score_micro = []
        f1_score_macro = []
        f1_score_weighted = []
        accuracy_ = []
        precision_ = []
        recall_ = []

        for i in range(1, 6):
            f1_score_micro.append(model[i]["f1_score_micro"])
            f1_score_macro.append(model[i]["f1_score_macro"])
            f1_score_weighted.append(model[i]["f1_score_weighted"])
            accuracy_.append(model[i]["accuracy"])
            precision_.append(model[i]["precision"])
            recall_.append(model[i]["recall"])
        
        f1_score_micro_mean = np.mean(f1_score_micro)
        f1_score_micro_std = np.std(f1_score_micro, ddof=1)
        f1_score_macro_mean = np.mean(f1_score_macro)
        f1_score_macro_std = np.std(f1_score_macro, ddof=1)
        f1_score_weighted_mean = np.mean(f1_score_weighted)
        f1_score_weighted_std = np.std(f1_score_weighted, ddof=1)
        accuracy_mean = np.mean(accuracy_)
        accuracy_std = np.std(accuracy_, ddof=1)
        precision_mean = np.mean(precision_)
        precision_std = np.std(precision_, ddof=1)
        recall_mean = np.mean(recall_)
        recall_std = np.std(recall_, ddof=1)

        stats[model["model_name"]] = {
            "f1_score_micro": f1_score_micro_mean, 
            "f1_score_micro_std": f1_score_micro_std,
            "f1_score_macro": f1_score_macro_mean,
            "f1_score_macro_std": f1_score_macro_std,
            "f1_score_weighted": f1_score_weighted_mean,
            "f1_score_weighted_std": f1_score_weighted_std,
            "accuracy": accuracy_mean,
            "accuracy_std": accuracy_std,
            "precision": precision_mean,
            "precision_std": precision_std,
            "recall": recall_mean,
            "recall_std": recall_std        
        }





    f1_score_micro = f1_score(Y_test_org, y_predict_test_org, average='micro')
    f1_score_macro = f1_score(Y_test_org, y_predict_test_org, average='macro')
    f1_score_weighted = f1_score(Y_test_org, y_predict_test_org, average='weighted')
    accuracy_ = accuracy_score(Y_test_org, y_predict_test_org)
    precision_ = precision_score(Y_test_org, y_predict_test_org)
    recall_ = recall_score(Y_test_org, y_predict_test_org)

    original = {
                    "f1_score_micro" : f1_score_micro,
                    "f1_score_micro_std" : 0,
                    "f1_score_macro" : f1_score_macro,
                    "f1_score_macro_std" : 0,
                    "f1_score_weighted" : f1_score_weighted,
                    "f1_score_weighted_std" : 0,
                    "accuracy" : accuracy_,
                    "accuracy_std" : 0,
                    "precision" : precision_,
                    "precision_std" : 0,
                    "recall" : recall_,
                    "recall_std" : 0
                }

    stats["original"] = original


    return stats






def mle_customer_travel(        
        org_test_df : pd.DataFrame,
        org_train_df : pd.DataFrame,
        ml_model,
        path_to_synthetic_data : str       
    ):

    """
    ml_model : should be provided like that - CatBoostClassifier()
    """
    
    le_frequent_flyer = LabelEncoder()
    le_annual_income_class = LabelEncoder()
    le_account_syncted = LabelEncoder()
    le_booked = LabelEncoder()

    column_name = "FrequentFlyer"
    org_train_df[column_name] = le_frequent_flyer.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_frequent_flyer.transform(org_test_df[column_name])

    column_name = "AnnualIncomeClass"
    org_train_df[column_name] = le_annual_income_class.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_annual_income_class.transform(org_test_df[column_name])

    column_name = "AccountSyncedToSocialMedia"
    org_train_df[column_name] = le_account_syncted.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_account_syncted.transform(org_test_df[column_name])

    column_name = "BookedHotelOrNot"
    org_train_df[column_name] = le_booked.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_booked.transform(org_test_df[column_name])

    X_train = org_train_df.drop("Target", axis=1)
    Y_train = org_train_df["Target"]

    X_test_org = org_test_df.drop("Target", axis=1)
    Y_test_org = org_test_df["Target"]

    ml = ml_model
    ml.fit(X_train, Y_train)
    y_predict_test_org= ml.predict(X_test_org)
    models_name = os.listdir(path_to_synthetic_data)



    outputs = []
    for model in models_name:

        model_output = {"model_name" : model}
        # print(model)

        for i in range(1, 6):
            # print(i)
            synt_test_df = pd.read_csv(f"{path_to_synthetic_data}/{model}/samples/test/sample{i}.csv", sep=",")[:len(X_test_org)]

            if "Unnamed: 0" in synt_test_df.columns:
                synt_test_df = synt_test_df.drop('Unnamed: 0', axis=1)

            column_name = "FrequentFlyer"
            synt_test_df[column_name] = le_frequent_flyer.transform(synt_test_df[column_name])
            column_name = "AnnualIncomeClass"
            synt_test_df[column_name] = le_annual_income_class.transform(synt_test_df[column_name])
            column_name = "AccountSyncedToSocialMedia"
            synt_test_df[column_name] = le_account_syncted.transform(synt_test_df[column_name])
            column_name = "BookedHotelOrNot"
            synt_test_df[column_name] = le_booked.transform(synt_test_df[column_name])

            X_test_synt = synt_test_df.drop("Target", axis=1)
            Y_test_synt = synt_test_df["Target"]

            y_predict_syn_test = ml.predict(X_test_synt)
            f1_score_micro = f1_score(Y_test_synt, y_predict_syn_test, average='micro')
            f1_score_macro = f1_score(Y_test_synt, y_predict_syn_test, average='macro')
            f1_score_weighted = f1_score(Y_test_synt, y_predict_syn_test, average='weighted')
            accuracy_ = accuracy_score(Y_test_synt, y_predict_syn_test)
            precision_ = precision_score(Y_test_synt, y_predict_syn_test)
            recall_ = recall_score(Y_test_synt, y_predict_syn_test)

            model_output[i] = {
                "f1_score_micro": f1_score_micro,
                "f1_score_macro": f1_score_macro,
                "f1_score_weighted": f1_score_weighted,
                "accuracy": accuracy_,
                "precision": precision_,
                "recall": recall_,
            }

        outputs.append(model_output)




    stats = {}

    for model in outputs:
        f1_score_micro = []
        f1_score_macro = []
        f1_score_weighted = []
        accuracy_ = []
        precision_ = []
        recall_ = []

        for i in range(1, 6):
            f1_score_micro.append(model[i]["f1_score_micro"])
            f1_score_macro.append(model[i]["f1_score_macro"])
            f1_score_weighted.append(model[i]["f1_score_weighted"])
            accuracy_.append(model[i]["accuracy"])
            precision_.append(model[i]["precision"])
            recall_.append(model[i]["recall"])
        
        f1_score_micro_mean = np.mean(f1_score_micro)
        f1_score_micro_std = np.std(f1_score_micro, ddof=1)
        f1_score_macro_mean = np.mean(f1_score_macro)
        f1_score_macro_std = np.std(f1_score_macro, ddof=1)
        f1_score_weighted_mean = np.mean(f1_score_weighted)
        f1_score_weighted_std = np.std(f1_score_weighted, ddof=1)
        accuracy_mean = np.mean(accuracy_)
        accuracy_std = np.std(accuracy_, ddof=1)
        precision_mean = np.mean(precision_)
        precision_std = np.std(precision_, ddof=1)
        recall_mean = np.mean(recall_)
        recall_std = np.std(recall_, ddof=1)

        stats[model["model_name"]] = {
            "f1_score_micro": f1_score_micro_mean, 
            "f1_score_micro_std": f1_score_micro_std,
            "f1_score_macro": f1_score_macro_mean,
            "f1_score_macro_std": f1_score_macro_std,
            "f1_score_weighted": f1_score_weighted_mean,
            "f1_score_weighted_std": f1_score_weighted_std,
            "accuracy": accuracy_mean,
            "accuracy_std": accuracy_std,
            "precision": precision_mean,
            "precision_std": precision_std,
            "recall": recall_mean,
            "recall_std": recall_std        
        }





    f1_score_micro = f1_score(Y_test_org, y_predict_test_org, average='micro')
    f1_score_macro = f1_score(Y_test_org, y_predict_test_org, average='macro')
    f1_score_weighted = f1_score(Y_test_org, y_predict_test_org, average='weighted')
    accuracy_ = accuracy_score(Y_test_org, y_predict_test_org)
    precision_ = precision_score(Y_test_org, y_predict_test_org)
    recall_ = recall_score(Y_test_org, y_predict_test_org)

    original = {
                    "f1_score_micro" : f1_score_micro,
                    "f1_score_micro_std" : 0,
                    "f1_score_macro" : f1_score_macro,
                    "f1_score_macro_std" : 0,
                    "f1_score_weighted" : f1_score_weighted,
                    "f1_score_weighted_std" : 0,
                    "accuracy" : accuracy_,
                    "accuracy_std" : 0,
                    "precision" : precision_,
                    "precision_std" : 0,
                    "recall" : recall_,
                    "recall_std" : 0
                }

    stats["original"] = original


    return stats




def mle_housing(
        org_test_df : pd.DataFrame,
        org_train_df : pd.DataFrame,
        ml_model,
        path_to_synthetic_data : str       
):
    
    le_ocean_proximity = LabelEncoder()

    column_name = "ocean_proximity"
    org_train_df[column_name] = le_ocean_proximity.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_ocean_proximity.transform(org_test_df[column_name])

    X_train = org_train_df.drop("median_house_value", axis=1)
    Y_train = org_train_df["median_house_value"]

    X_test_org = org_test_df.drop("median_house_value", axis=1)
    Y_test_org = org_test_df["median_house_value"]

    ml = ml_model
    ml.fit(X_train, Y_train)

    y_predict_test_org= ml.predict(X_test_org)
    models_name = os.listdir(path_to_synthetic_data)

    outputs = []

    for model in models_name:

        model_output = {"model_name" : model}
        # print(model)

        for i in range(1, 6):
            # print(i)
            synt_test_df = pd.read_csv(f"{path_to_synthetic_data}/{model}/samples/test/sample{i}.csv", sep=",")[:len(X_test_org)]

            if "Unnamed: 0" in synt_test_df.columns:
                synt_test_df = synt_test_df.drop('Unnamed: 0', axis=1)

            column_name = "ocean_proximity"
            synt_test_df[column_name] = le_ocean_proximity.transform(synt_test_df[column_name])

            X_test_synt = synt_test_df.drop("median_house_value", axis=1)
            Y_test_synt = synt_test_df["median_house_value"]

            y_predict_syn_test = ml.predict(X_test_synt)
            mae = mean_absolute_error(Y_test_synt, y_predict_syn_test)
            mse = mean_squared_error(Y_test_synt, y_predict_syn_test)
            rmse = np.sqrt(mse)
            r2 = r2_score(Y_test_synt, y_predict_syn_test)
            mape = mean_absolute_percentage_error(Y_test_synt, y_predict_syn_test)

            model_output[i] = {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "mape": mape
            }

        outputs.append(model_output)

    stats = {}

    for model in outputs:
        mae = []
        mse = []
        rmse = []
        r2 = []
        mape = []

        for i in range(1, 6):
            mae.append(model[i]["mae"])
            mse.append(model[i]["mse"])
            rmse.append(model[i]["rmse"])
            r2.append(model[i]["r2"])
            mape.append(model[i]["mape"])
        
        mae_mean = np.mean(mae)
        mae_std = np.std(mae, ddof=1)
        mse_mean = np.mean(mse)
        mse_std = np.std(mse, ddof=1)
        rmse_mean = np.mean(rmse)
        rmse_std = np.std(rmse, ddof=1)
        r2_mean = np.mean(r2)
        r2_std = np.std(r2, ddof=1)
        mape_mean = np.mean(mape)
        mape_std = np.std(mape, ddof=1)

        stats[model["model_name"]] = {
            "mae": mae_mean, 
            "mae_std": mae_std,
            "mse": mse_mean,
            "mse_std": mse_std,
            "rmse": rmse_mean,
            "rmse_std": rmse_std,
            "r2": r2_mean,
            "r2_std": r2_std,
            "mape": mape_mean,
            "mape_std": mape_std        
        }

    mae = mean_absolute_error(Y_test_org, y_predict_test_org)
    mse = mean_squared_error(Y_test_org, y_predict_test_org)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test_org, y_predict_test_org)
    mape = mean_absolute_percentage_error(Y_test_org, y_predict_test_org)

    original = {
                    "mae" : mae,
                    "mae_std" : 0,
                    "mse" : mse,
                    "mse_std" : 0,
                    "rmse" : rmse,
                    "rmse_std" : 0,
                    "r2" : r2,
                    "r2_std" : 0,
                    "mape" : mape,
                    "mape_std" : 0
                }

    stats["original"] = original

    return stats


    



def mle_stroke_healthcare(        
        org_test_df : pd.DataFrame,
        org_train_df : pd.DataFrame,
        ml_model,
        path_to_synthetic_data : str       
    ):

    """
    ml_model : should be provided like that - CatBoostClassifier()
    """
    
    le_gender = LabelEncoder()
    le_ever_married = LabelEncoder()
    le_work_type = LabelEncoder()
    le_residence_type = LabelEncoder()
    le_smoking_status = LabelEncoder()

    column_name = "gender"
    org_train_df[column_name] = le_gender.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_gender.transform(org_test_df[column_name])

    column_name = "ever_married"
    org_train_df[column_name] = le_ever_married.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_ever_married.transform(org_test_df[column_name])

    column_name = "work_type"
    org_train_df[column_name] = le_work_type.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_work_type.transform(org_test_df[column_name])

    column_name = "Residence_type"
    org_train_df[column_name] = le_residence_type.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_residence_type.transform(org_test_df[column_name])

    column_name = "smoking_status"
    org_train_df[column_name] = le_smoking_status.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_smoking_status.transform(org_test_df[column_name])

    X_train = org_train_df.drop("stroke", axis=1)
    Y_train = org_train_df["stroke"]

    X_test_org = org_test_df.drop("stroke", axis=1)
    Y_test_org = org_test_df["stroke"]

    ml = ml_model
    ml.fit(X_train, Y_train)
    y_predict_test_org= ml.predict(X_test_org)
    models_name = os.listdir(path_to_synthetic_data)



    outputs = []
    for model in models_name:

        model_output = {"model_name" : model}
        # print(model)

        for i in range(1, 6):
            # print(i)
            synt_test_df = pd.read_csv(f"{path_to_synthetic_data}/{model}/samples/test/sample{i}.csv", sep=",")[:len(X_test_org)]

            if "Unnamed: 0" in synt_test_df.columns:
                synt_test_df = synt_test_df.drop('Unnamed: 0', axis=1)

            column_name = "gender"
            synt_test_df[column_name] = le_gender.transform(synt_test_df[column_name])
            column_name = "ever_married"
            synt_test_df[column_name] = le_ever_married.transform(synt_test_df[column_name])
            column_name = "work_type"
            synt_test_df[column_name] = le_work_type.transform(synt_test_df[column_name])
            column_name = "Residence_type"
            synt_test_df[column_name] = le_residence_type.transform(synt_test_df[column_name])
            column_name = "smoking_status"
            synt_test_df[column_name] = le_smoking_status.transform(synt_test_df[column_name])

            X_test_synt = synt_test_df.drop("stroke", axis=1)
            Y_test_synt = synt_test_df["stroke"]

            y_predict_syn_test = ml.predict(X_test_synt)
            f1_score_micro = f1_score(Y_test_synt, y_predict_syn_test, average='micro')
            f1_score_macro = f1_score(Y_test_synt, y_predict_syn_test, average='macro')
            f1_score_weighted = f1_score(Y_test_synt, y_predict_syn_test, average='weighted')
            accuracy_ = accuracy_score(Y_test_synt, y_predict_syn_test)
            precision_ = precision_score(Y_test_synt, y_predict_syn_test)
            recall_ = recall_score(Y_test_synt, y_predict_syn_test)

            model_output[i] = {
                "f1_score_micro": f1_score_micro,
                "f1_score_macro": f1_score_macro,
                "f1_score_weighted": f1_score_weighted,
                "accuracy": accuracy_,
                "precision": precision_,
                "recall": recall_,
            }

        outputs.append(model_output)




    stats = {}

    for model in outputs:
        f1_score_micro = []
        f1_score_macro = []
        f1_score_weighted = []
        accuracy_ = []
        precision_ = []
        recall_ = []

        for i in range(1, 6):
            f1_score_micro.append(model[i]["f1_score_micro"])
            f1_score_macro.append(model[i]["f1_score_macro"])
            f1_score_weighted.append(model[i]["f1_score_weighted"])
            accuracy_.append(model[i]["accuracy"])
            precision_.append(model[i]["precision"])
            recall_.append(model[i]["recall"])
        
        f1_score_micro_mean = np.mean(f1_score_micro)
        f1_score_micro_std = np.std(f1_score_micro, ddof=1)
        f1_score_macro_mean = np.mean(f1_score_macro)
        f1_score_macro_std = np.std(f1_score_macro, ddof=1)
        f1_score_weighted_mean = np.mean(f1_score_weighted)
        f1_score_weighted_std = np.std(f1_score_weighted, ddof=1)
        accuracy_mean = np.mean(accuracy_)
        accuracy_std = np.std(accuracy_, ddof=1)
        precision_mean = np.mean(precision_)
        precision_std = np.std(precision_, ddof=1)
        recall_mean = np.mean(recall_)
        recall_std = np.std(recall_, ddof=1)

        stats[model["model_name"]] = {
            "f1_score_micro": f1_score_micro_mean, 
            "f1_score_micro_std": f1_score_micro_std,
            "f1_score_macro": f1_score_macro_mean,
            "f1_score_macro_std": f1_score_macro_std,
            "f1_score_weighted": f1_score_weighted_mean,
            "f1_score_weighted_std": f1_score_weighted_std,
            "accuracy": accuracy_mean,
            "accuracy_std": accuracy_std,
            "precision": precision_mean,
            "precision_std": precision_std,
            "recall": recall_mean,
            "recall_std": recall_std        
        }





    f1_score_micro = f1_score(Y_test_org, y_predict_test_org, average='micro')
    f1_score_macro = f1_score(Y_test_org, y_predict_test_org, average='macro')
    f1_score_weighted = f1_score(Y_test_org, y_predict_test_org, average='weighted')
    accuracy_ = accuracy_score(Y_test_org, y_predict_test_org)
    precision_ = precision_score(Y_test_org, y_predict_test_org)
    recall_ = recall_score(Y_test_org, y_predict_test_org)

    original = {
                    "f1_score_micro" : f1_score_micro,
                    "f1_score_micro_std" : 0,
                    "f1_score_macro" : f1_score_macro,
                    "f1_score_macro_std" : 0,
                    "f1_score_weighted" : f1_score_weighted,
                    "f1_score_weighted_std" : 0,
                    "accuracy" : accuracy_,
                    "accuracy_std" : 0,
                    "precision" : precision_,
                    "precision_std" : 0,
                    "recall" : recall_,
                    "recall_std" : 0
                }

    stats["original"] = original


    return stats




