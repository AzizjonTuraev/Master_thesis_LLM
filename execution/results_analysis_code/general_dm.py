import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder

import os


def dm_adult(
        org_train_df : pd.DataFrame,
        org_test_df : pd.DataFrame,
        ml_model,
        path_to_synthetic_dataset : str,

):

    models_name = os.listdir(path_to_synthetic_dataset)
    outputs = []
    org_train_df["original"] = [1]*len(org_train_df)
    org_test_df["original"] = [1]*len(org_test_df)


    for model in models_name:

        print(model)
        model_output = {"model_name" : model}
        for i in range(1, 6):

            print(i)
            synt_test_df = pd.read_csv(f"{path_to_synthetic_dataset}/{model}/samples/test/sample{i}.csv", sep=",")[:len(org_test_df)]
            synt_train_df = pd.read_csv(f"{path_to_synthetic_dataset}/{model}/samples/train/sample{i}.csv", sep=",")[:len(org_train_df)]        

            if "Unnamed: 0" in synt_test_df.columns:
                synt_test_df = synt_test_df.drop('Unnamed: 0', axis=1)

            if "Unnamed: 0" in synt_train_df.columns:
                synt_train_df = synt_train_df.drop('Unnamed: 0', axis=1)

            synt_test_df["original"] = [0]*len(synt_test_df)
            synt_train_df["original"] = [0]*len(synt_train_df)

            train_set = pd.concat([org_train_df, synt_train_df], ignore_index=True)
            test_set = pd.concat([org_test_df, synt_test_df], ignore_index=True)

            le_workclass = LabelEncoder()
            le_marital_status = LabelEncoder()
            le_occupation = LabelEncoder()
            le_race = LabelEncoder()
            le_gender = LabelEncoder()
            le_native_country = LabelEncoder()
            le_income = LabelEncoder()

            column_name = "workclass"
            train_set[column_name] = le_workclass.fit_transform(train_set[column_name])
            test_set[column_name] = le_workclass.transform(test_set[column_name])

            column_name = "marital-status"
            train_set[column_name] = le_marital_status.fit_transform(train_set[column_name])
            test_set[column_name] = le_marital_status.transform(test_set[column_name])

            column_name = "occupation"
            train_set[column_name] = le_occupation.fit_transform(train_set[column_name])
            test_set[column_name] = le_occupation.transform(test_set[column_name])

            column_name = "race"
            train_set[column_name] = le_race.fit_transform(train_set[column_name])
            test_set[column_name] = le_race.transform(test_set[column_name])

            column_name = "gender"
            train_set[column_name] = le_gender.fit_transform(train_set[column_name])
            test_set[column_name] = le_gender.transform(test_set[column_name])

            column_name = "native-country"
            train_set[column_name] = le_native_country.fit_transform(train_set[column_name])
            test_set[column_name] = le_native_country.transform(test_set[column_name])

            column_name = "income"
            train_set[column_name] = le_income.fit_transform(train_set[column_name])
            test_set[column_name] = le_income.transform(test_set[column_name])

            X_train = train_set.drop("original", axis=1)
            Y_train = train_set["original"]
            X_test = test_set.drop("original", axis=1)
            Y_test = test_set["original"]

            ml = ml_model
            ml.fit(X_train, Y_train)
            y_predict = ml.predict(X_test)

            accuracy_score_ = accuracy_score(Y_test, y_predict)

            model_output[i] = {
                "accuracy": accuracy_score_,
            }

        outputs.append(model_output)



    stats = {}

    for model in outputs:
        accuracy_ = []

        for i in range(1, 6):
            accuracy_.append(model[i]["accuracy"])
        
        accuracy_mean = np.mean(accuracy_)
        accuracy_std = np.std(accuracy_, ddof=1)

        stats[model["model_name"]] = {
            "accuracy": accuracy_mean,
            "accuracy_std": accuracy_std,
        }

    return stats



def dm_customer_travel(
        org_train_df : pd.DataFrame,
        org_test_df : pd.DataFrame,
        ml_model,
        path_to_synthetic_dataset : str,

):

    models_name = os.listdir(path_to_synthetic_dataset)
    outputs = []
    org_train_df["original"] = [1]*len(org_train_df)
    org_test_df["original"] = [1]*len(org_test_df)


    for model in models_name:

        print(model)
        model_output = {"model_name" : model}
        for i in range(1, 6):

            print(i)
            synt_test_df = pd.read_csv(f"{path_to_synthetic_dataset}/{model}/samples/test/sample{i}.csv", sep=",")[:len(org_test_df)]
            synt_train_df = pd.read_csv(f"{path_to_synthetic_dataset}/{model}/samples/train/sample{i}.csv", sep=",")[:len(org_train_df)]        

            if "Unnamed: 0" in synt_test_df.columns:
                synt_test_df = synt_test_df.drop('Unnamed: 0', axis=1)

            if "Unnamed: 0" in synt_train_df.columns:
                synt_train_df = synt_train_df.drop('Unnamed: 0', axis=1)

            synt_test_df["original"] = [0]*len(synt_test_df)
            synt_train_df["original"] = [0]*len(synt_train_df)

            train_set = pd.concat([org_train_df, synt_train_df], ignore_index=True)
            test_set = pd.concat([org_test_df, synt_test_df], ignore_index=True)

            le_frequent_flyer = LabelEncoder()
            le_annual_income_class = LabelEncoder()
            le_account_syncted = LabelEncoder()
            le_booked = LabelEncoder()

            column_name = "FrequentFlyer"
            train_set[column_name] = le_frequent_flyer.fit_transform(train_set[column_name])
            test_set[column_name] = le_frequent_flyer.transform(test_set[column_name])

            column_name = "AnnualIncomeClass"
            train_set[column_name] = le_annual_income_class.fit_transform(train_set[column_name])
            test_set[column_name] = le_annual_income_class.transform(test_set[column_name])

            column_name = "AccountSyncedToSocialMedia"
            train_set[column_name] = le_account_syncted.fit_transform(train_set[column_name])
            test_set[column_name] = le_account_syncted.transform(test_set[column_name])

            column_name = "BookedHotelOrNot"
            train_set[column_name] = le_booked.fit_transform(train_set[column_name])
            test_set[column_name] = le_booked.transform(test_set[column_name])

            X_train = train_set.drop("original", axis=1)
            Y_train = train_set["original"]
            X_test = test_set.drop("original", axis=1)
            Y_test = test_set["original"]

            ml = ml_model
            ml.fit(X_train, Y_train)
            y_predict = ml.predict(X_test)

            accuracy_score_ = accuracy_score(Y_test, y_predict)

            model_output[i] = {
                "accuracy": accuracy_score_,
            }

        outputs.append(model_output)



    stats = {}

    for model in outputs:
        accuracy_ = []

        for i in range(1, 6):
            accuracy_.append(model[i]["accuracy"])
        
        accuracy_mean = np.mean(accuracy_)
        accuracy_std = np.std(accuracy_, ddof=1)

        stats[model["model_name"]] = {
            "accuracy": accuracy_mean,
            "accuracy_std": accuracy_std,
        }

    return stats




def dm_housing(
        org_train_df : pd.DataFrame,
        org_test_df : pd.DataFrame,
        ml_model,
        path_to_synthetic_dataset : str,

):

    models_name = os.listdir(path_to_synthetic_dataset)
    outputs = []
    org_train_df["original"] = [1]*len(org_train_df)
    org_test_df["original"] = [1]*len(org_test_df)


    for model in models_name:

        print(model)
        model_output = {"model_name" : model}
        for i in range(1, 6):

            print(i)
            synt_test_df = pd.read_csv(f"{path_to_synthetic_dataset}/{model}/samples/test/sample{i}.csv", sep=",")[:len(org_test_df)]
            synt_train_df = pd.read_csv(f"{path_to_synthetic_dataset}/{model}/samples/train/sample{i}.csv", sep=",")[:len(org_train_df)]        

            if "Unnamed: 0" in synt_test_df.columns:
                synt_test_df = synt_test_df.drop('Unnamed: 0', axis=1)

            if "Unnamed: 0" in synt_train_df.columns:
                synt_train_df = synt_train_df.drop('Unnamed: 0', axis=1)

            synt_test_df["original"] = [0]*len(synt_test_df)
            synt_train_df["original"] = [0]*len(synt_train_df)

            train_set = pd.concat([org_train_df, synt_train_df], ignore_index=True)
            test_set = pd.concat([org_test_df, synt_test_df], ignore_index=True)

            le_ocean_proximity = LabelEncoder()

            column_name = "ocean_proximity"
            train_set[column_name] = le_ocean_proximity.fit_transform(train_set[column_name])
            test_set[column_name] = le_ocean_proximity.transform(test_set[column_name])

            X_train = train_set.drop("original", axis=1)
            Y_train = train_set["original"]
            X_test = test_set.drop("original", axis=1)
            Y_test = test_set["original"]

            ml = ml_model
            ml.fit(X_train, Y_train)
            y_predict = ml.predict(X_test)

            accuracy_score_ = accuracy_score(Y_test, y_predict)

            model_output[i] = {
                "accuracy": accuracy_score_,
            }

        outputs.append(model_output)


    stats = {}

    for model in outputs:
        accuracy_ = []

        for i in range(1, 6):
            accuracy_.append(model[i]["accuracy"])
        
        accuracy_mean = np.mean(accuracy_)
        accuracy_std = np.std(accuracy_, ddof=1)

        stats[model["model_name"]] = {
            "accuracy": accuracy_mean,
            "accuracy_std": accuracy_std,
        }

    return stats




def dm_stroke_healthcare(
        org_train_df : pd.DataFrame,
        org_test_df : pd.DataFrame,
        ml_model,
        path_to_synthetic_dataset : str,
):

    models_name = os.listdir(path_to_synthetic_dataset)
    outputs = []
    org_train_df["original"] = [1]*len(org_train_df)
    org_test_df["original"] = [1]*len(org_test_df)


    for model in models_name:

        print(model)
        model_output = {"model_name" : model}
        for i in range(1, 6):

            print(i)
            synt_test_df = pd.read_csv(f"{path_to_synthetic_dataset}/{model}/samples/test/sample{i}.csv", sep=",")[:len(org_test_df)]
            synt_train_df = pd.read_csv(f"{path_to_synthetic_dataset}/{model}/samples/train/sample{i}.csv", sep=",")[:len(org_train_df)]        

            if "Unnamed: 0" in synt_test_df.columns:
                synt_test_df = synt_test_df.drop('Unnamed: 0', axis=1)

            if "Unnamed: 0" in synt_train_df.columns:
                synt_train_df = synt_train_df.drop('Unnamed: 0', axis=1)

            synt_test_df["original"] = [0]*len(synt_test_df)
            synt_train_df["original"] = [0]*len(synt_train_df)

            train_set = pd.concat([org_train_df, synt_train_df], ignore_index=True)
            test_set = pd.concat([org_test_df, synt_test_df], ignore_index=True)

            le_gender = LabelEncoder()
            le_ever_married = LabelEncoder()
            le_work_type = LabelEncoder()
            le_residence_type = LabelEncoder()
            le_smoking_status = LabelEncoder()

            column_name = "gender"
            train_set[column_name] = le_gender.fit_transform(train_set[column_name])
            test_set[column_name] = le_gender.transform(test_set[column_name])

            column_name = "ever_married"
            train_set[column_name] = le_ever_married.fit_transform(train_set[column_name])
            test_set[column_name] = le_ever_married.transform(test_set[column_name])

            column_name = "work_type"
            train_set[column_name] = le_work_type.fit_transform(train_set[column_name])
            test_set[column_name] = le_work_type.transform(test_set[column_name])

            column_name = "Residence_type"
            train_set[column_name] = le_residence_type.fit_transform(train_set[column_name])
            test_set[column_name] = le_residence_type.transform(test_set[column_name])

            column_name = "smoking_status"
            train_set[column_name] = le_smoking_status.fit_transform(train_set[column_name])
            test_set[column_name] = le_smoking_status.transform(test_set[column_name])

            X_train = train_set.drop("original", axis=1)
            Y_train = train_set["original"]
            X_test = test_set.drop("original", axis=1)
            Y_test = test_set["original"]

            ml = ml_model
            ml.fit(X_train, Y_train)
            y_predict = ml.predict(X_test)

            accuracy_score_ = accuracy_score(Y_test, y_predict)

            model_output[i] = {
                "accuracy": accuracy_score_,
            }

        outputs.append(model_output)


    stats = {}

    for model in outputs:
        accuracy_ = []

        for i in range(1, 6):
            accuracy_.append(model[i]["accuracy"])
        
        accuracy_mean = np.mean(accuracy_)
        accuracy_std = np.std(accuracy_, ddof=1)

        stats[model["model_name"]] = {
            "accuracy": accuracy_mean,
            "accuracy_std": accuracy_std,
        }

    return stats



