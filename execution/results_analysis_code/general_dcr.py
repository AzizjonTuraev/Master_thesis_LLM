import pandas as pd
import numpy as np

from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import os

from sklearn.preprocessing import StandardScaler


def compute_dcr(synthetic_data, original_data):
    """
    Compute the Distance to Closest Record (DCR) for each synthetic sample.
    
    Parameters:
    - synthetic_data: array-like, shape (n_samples, n_features), synthetic samples
    - original_data: array-like, shape (m_samples, n_features), original samples
    
    Returns:
    - dcr_scores: array of minimum distances for each synthetic sample
    """
    # Compute L1 distance between each synthetic record and all original records
    distances = cdist(synthetic_data, original_data, metric='cityblock')
    
    # Get the minimum distance to closest original record for each synthetic record
    dcr_scores = distances.min(axis=1)
    
    return dcr_scores


def plot_and_save(dcr_score, bins, filter_limit_x, folder_to_save_plots, model):

    filtered_scores = dcr_score[dcr_score < filter_limit_x]  # Adjust threshold as needed
    plt.hist(filtered_scores, bins=bins, color='skyblue', edgecolor='black', density=True)
    plt.xlabel('DCR')
    plt.ylabel('Density')
#    plt.ylim(top=0.35) # stroke healthcare
    # plt.ylim(top=0.045) # adult
    # plt.ylim(top=7) # customer no top limit
    plt.title(model)
    plt.savefig(f"{folder_to_save_plots}/{model}.png")
    plt.show()



def dcr_adult(
        org_train_df : pd.DataFrame,
        org_test_df : pd.DataFrame,
        path_to_synthetic_dataset : str,
        filter_limit_x : int = 100,
        bins : int = 100

):
    folder_to_save_plots = f"../final_results/{path_to_synthetic_dataset}_dcr_plots"
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

    os.makedirs(folder_to_save_plots, exist_ok=True)
    dcr_score = compute_dcr(org_test_df, org_train_df)
    model = "Original Test Data Set"
    plot_and_save(dcr_score, bins, filter_limit_x, folder_to_save_plots, model)

    path_to_synthetic_dataset = f"../generated_datasets/{path_to_synthetic_dataset}"
    models_name = os.listdir(path_to_synthetic_dataset)

    for model in models_name:

        print(model)
        i = 1

        synt_test_df = pd.read_csv(f"{path_to_synthetic_dataset}/{model}/samples/train/sample{i}.csv", sep=",")[:len(org_test_df)]

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

        dcr_score = compute_dcr(synt_test_df, org_train_df)
        plot_and_save(dcr_score, bins, filter_limit_x, folder_to_save_plots, model)
        



def dcr_customer_travel(
        org_train_df : pd.DataFrame,
        org_test_df : pd.DataFrame,
        path_to_synthetic_dataset : str,
        filter_limit_x : int = 1000,
        bins : int = 10

):
    folder_to_save_plots = f"../final_results/{path_to_synthetic_dataset}_dcr_plots"

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

    # Step 2: Standardize both datasets
    scaler = StandardScaler()
    org_train_df = scaler.fit_transform(org_train_df)
    org_test_df = scaler.transform(org_test_df)

    os.makedirs(folder_to_save_plots, exist_ok=True)
    dcr_score = compute_dcr(org_test_df, org_train_df)
    model = "Original Test Data Set"
    plot_and_save(dcr_score, bins, filter_limit_x, 
                folder_to_save_plots, model)

    path_to_synthetic_dataset = f"../generated_datasets/{path_to_synthetic_dataset}"
    models_name = os.listdir(path_to_synthetic_dataset)

    for model in models_name:

        print(model)
        i = 1

        synt_test_df = pd.read_csv(f"{path_to_synthetic_dataset}/{model}/samples/train/sample{i}.csv", sep=",")[:len(org_test_df)]

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

        synt_test_df = scaler.transform(synt_test_df)
        dcr_score = compute_dcr(synt_test_df, org_train_df)
        plot_and_save(dcr_score, bins, filter_limit_x, 
                    folder_to_save_plots, model)
        




def dcr_housing(
        org_train_df : pd.DataFrame,
        org_test_df : pd.DataFrame,
        path_to_synthetic_dataset : str,
        filter_limit_x : int = 5000,
        bins : int = 100

):
    folder_to_save_plots = f"../final_results/{path_to_synthetic_dataset}_dcr_plots"

    le_ocean_proximity = LabelEncoder()

    column_name = "ocean_proximity"
    
    org_train_df[column_name] = le_ocean_proximity.fit_transform(org_train_df[column_name])
    org_test_df[column_name] = le_ocean_proximity.transform(org_test_df[column_name])

    os.makedirs(folder_to_save_plots, exist_ok=True)
    dcr_score = compute_dcr(org_test_df, org_train_df)
    model = "Original Test Data Set"
    plot_and_save(dcr_score, bins, filter_limit_x, folder_to_save_plots, model)

    path_to_synthetic_dataset = f"../generated_datasets/{path_to_synthetic_dataset}"
    models_name = os.listdir(path_to_synthetic_dataset)

    for model in models_name:

        print(model)
        i = 1

        synt_test_df = pd.read_csv(f"{path_to_synthetic_dataset}/{model}/samples/train/sample{i}.csv", sep=",")[:len(org_test_df)]

        if "Unnamed: 0" in synt_test_df.columns:
            synt_test_df = synt_test_df.drop('Unnamed: 0', axis=1)

        column_name = "ocean_proximity"
        synt_test_df[column_name] = le_ocean_proximity.transform(synt_test_df[column_name])

        dcr_score = compute_dcr(synt_test_df, org_train_df)
        plot_and_save(dcr_score, bins, filter_limit_x, folder_to_save_plots, model)
        


def dcr_stroke_healthcare(
        org_train_df : pd.DataFrame,
        org_test_df : pd.DataFrame,
        path_to_synthetic_dataset : str,
        filter_limit_x : int = 5000,
        bins : int = 100

):
    folder_to_save_plots = f"../final_results/{path_to_synthetic_dataset}_dcr_plots"

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

    os.makedirs(folder_to_save_plots, exist_ok=True)
    dcr_score = compute_dcr(org_test_df, org_train_df)
    model = "Original Test Data Set"
    plot_and_save(dcr_score, bins, filter_limit_x, folder_to_save_plots, model)

    path_to_synthetic_dataset = f"../generated_datasets/{path_to_synthetic_dataset}"
    models_name = os.listdir(path_to_synthetic_dataset)

    for model in models_name:

        print(model)
        i = 1

        synt_test_df = pd.read_csv(f"{path_to_synthetic_dataset}/{model}/samples/train/sample{i}.csv", sep=",")[:len(org_test_df)]

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


        dcr_score = compute_dcr(synt_test_df, org_train_df)
        plot_and_save(dcr_score, bins, filter_limit_x, folder_to_save_plots, model)
        



