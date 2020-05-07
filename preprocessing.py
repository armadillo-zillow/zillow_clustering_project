import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

def impute_regionidcity(train, validate, test):
    """
    This function does the following:
    1. Takes in the train, validate, and test datasets
    2. Creates the KNNImputer object
    3. Fits the object to the regionidcity feature in the train dataset
    4. Transforms the regionidcity feature in the train, validate, and test datasets
    """

    imputer = KNNImputer(n_neighbors=5)
    imputer.fit(train[["regionidcity"]])
    train["regionidcity"] = imputer.transform(train[["regionidcity"]])
    validate["regionidcity"] = imputer.transform(validate[["regionidcity"]])
    test["regionidcity"] = imputer.transform(test[["regionidcity"]])

    return imputer, train, validate, test

def convert_dtypes(df, columns=[], dtype="object"):
    """
    Function does the follwing:
    1. Takes in a DataFrame, specified columns to transform as a list, and the desired data type of the specified columns as a string
    2. Returns a DataFrame with columns converted to specified dtype
    """

    # convert some numeric variables that are categorical by nature so that they do not end up scaled
    for col in columns:
        df[col] = df[col].astype(dtype)

    return df

def scale_numeric_data(train, validate, test):
    """
    Docstring
    """

    # convert some numeric variables that are categorical by nature into objects so that they do not end up scaled
    train = convert_dtypes(train, columns=["fips", "regionidcity", "regionidcounty", "regionidzip"], dtype="object")

    # creating a list of features whose data type is number
    numeric_columns = train.select_dtypes("number").columns.tolist()

    # scale numeric features
    scaler = MinMaxScaler()
    scaler.fit(train[numeric_columns])
    train[numeric_columns] = scaler.transform(train[numeric_columns])
    validate[numeric_columns] = scaler.transform(validate[numeric_columns])
    test[numeric_columns] = scaler.transform(test[numeric_columns])

    return scaler, train, validate, test

def split_impute_scale(df):
    """
    Docstring
    """

    # split data into train, validate, test
    train, test = train_test_split(df, train_size=.8, random_state=56, stratify=df.county)
    train, validate = train_test_split(train, train_size=.75, random_state=56, stratify=train.county)

    # KNN imputation for regionidcity
    imputer, train, validate, test = impute_regionidcity(train, validate, test)

    # scale numeric data
    scaler, train, validate, test = scale_numeric_data(train, validate, test)

    return imputer, scaler, train, validate, test