import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

from os import path

from env import host, user, password

def get_db_url(dbname) -> str:
    url = 'mysql+pymysql://{}:{}@{}/{}'
    return url.format(user, password, host, dbname)

def get_zillow_data():
    """
    Docstring
    """
    # query zillow data
    zillow_query = """
    SELECT prop.*, pred.logerror, pred.transactiondate, ac.airconditioningdesc, ar.architecturalstyledesc, bu.buildingclassdesc, he.heatingorsystemdesc, la.propertylandusedesc, st.storydesc, co.typeconstructiondesc
    FROM properties_2017 AS prop
    JOIN predictions_2017 AS pred USING(parcelid)
    LEFT JOIN airconditioningtype AS ac USING(airconditioningtypeid)
    LEFT JOIN architecturalstyletype AS ar USING(architecturalstyletypeid)
    LEFT JOIN buildingclasstype AS bu USING(buildingclasstypeid)
    LEFT JOIN heatingorsystemtype AS he USING(heatingorsystemtypeid)
    LEFT JOIN propertylandusetype AS la USING(propertylandusetypeid)
    LEFT JOIN storytype AS st USING(storytypeid)
    LEFT JOIN typeconstructiontype as co USING(typeconstructiontypeid)
    WHERE prop.latitude IS NOT NULL
    AND prop.longitude IS NOT NULL;"""
    
    # get database url
    zillow_url = get_db_url("zillow")
    
    # get pandas to read sql query + db url and return a DataFrame if the .csv of the data doesn't already exist
    if path.exists("zillow_clustering.csv"):
        df = pd.read_csv("zillow_clustering.csv", index_col=0)
    else:
        df = pd.read_sql(zillow_query, zillow_url)
        df.to_csv("zillow_clustering.csv")
    
    return df

def nulls_by_col(df):
    """
    Docstring
    """
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    pct_missing = num_missing / rows
    cols_missing = pd.DataFrame({
        "num_rows_missing": num_missing,
        "pct_rows_missing": pct_missing
    })
    return cols_missing

def nulls_by_row(df):
    """
    Docstring
    """
    num_cols_missing = df.isnull().sum(axis=1)
    pct_cols_missing = df.isnull().sum(axis=1) / df.shape[1]
    rows_missing = pd.DataFrame({
        'num_cols_missing': num_cols_missing,
        'pct_cols_missing': pct_cols_missing
    }).reset_index().groupby(['num_cols_missing','pct_cols_missing']).count().rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing

def remove_columns(df, cols_to_remove):
    """
    Docstring
    """
    df = df.drop(columns=cols_to_remove)
    return df

def handle_missing_values(df, prop_required_column = .60, prop_required_row = .60):
    """
    Docstring
    """
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def convert_dtypes(df):
    """
    Function does the follwing:
    1. Takes in a DataFrame
    2. Returns a DataFrame with some of the initally numeric variables converted to strings as they are more categorical by nature.
    """
    # convert some numeric variables that are categorical by nature so that they do not end up scaled
    for col in ['fips', 'regionidcity', 'regionidcounty', 'regionidzip']:
        df[col] = df[col].astype('object')
    return df

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

def scale_numeric_data(train, validate, test):
    """
    Docstring
    """
    # convert some numeric variables that are categorical by nature so that they do not end up scaled
    for col in ['fips', 'regionidcity', 'regionidcounty', 'regionidzip']:
        train[col] = train[col].astype('object')
    
    # creating a list of features whose data type is number
    numeric_columns = train.select_dtypes("number").columns.tolist()

    # scale numeric features
    scaler = MinMaxScaler()
    scaler.fit(train[numeric_columns])
    train[numeric_columns] = scaler.transform(train[numeric_columns])
    validate[numeric_columns] = scaler.transform(validate[numeric_columns])
    test[numeric_columns] = scaler.transform(test[numeric_columns])

    return scaler, train, validate, test

def prep_zillow(df):
    """
    Docstring
    """

    # removing any properties that are likely to be something other than single unit properties
    df = df[df.propertylandusetypeid.isin([260, 261, 262, 279])]
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0)]

    # unitcnt imputation
    df.unitcnt = df.unitcnt.fillna(1.0)

    # filter out units with more than one unit
    df = df[df.unitcnt == 1.0]

    # dropping unnecessary columns
    df.drop(columns="id", inplace=True)

    # calling handle_missing_values to remove columns and rows that do not meet the default threshold critera necessary to retain variable or index
    df = handle_missing_values(df)

    # dropping id columns
    df = df.drop(columns=["propertylandusetypeid", "heatingorsystemtypeid"])

    # heatingorsystemdesc imputation with "None" as properties are in SoCal so "None" is reasonable
    df.heatingorsystemdesc = df.heatingorsystemdesc.fillna("None")

    # dropping propertyzoningdesc we have already filtered the data to single unit properties
    df = df.drop(columns="propertyzoningdesc")

    # buildingqualitytypeid imputation
    df.buildingqualitytypeid = df.buildingqualitytypeid.fillna(df.buildingqualitytypeid.median())

    # lotsizesquarefeet imputation with mode
    df.lotsizesquarefeet = df.lotsizesquarefeet.fillna(6000.0)

    # calculatedbathnbr imputation
    df.calculatedbathnbr = df.calculatedbathnbr.fillna(df.calculatedbathnbr.median())

    # calculatedfinishedsquarefeet imputation with mode
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.fillna(1120.0)

    # drop finishedsquarefeet12 as the info appears to be redundant when compared to calculatedfinishedsquarefeet
    df.drop(columns="finishedsquarefeet12", inplace=True)

    # fullbathcnt imputation
    df.fullbathcnt = df.fullbathcnt.fillna(df.fullbathcnt.median())

    # yearbuilt
    df.yearbuilt = df.yearbuilt.fillna(round(df.yearbuilt.mean()))

    # structuretaxvaluedollarcnt imputation based on the difference between taxvaluedollarcnt and landtaxvaluedollarcnt as 
    # 99.9% of the values in structuretaxvaluedollarcnt are equal to this difference
    df.structuretaxvaluedollarcnt = df.structuretaxvaluedollarcnt.fillna(df.taxvaluedollarcnt - df.landtaxvaluedollarcnt)

    # drop the remaining row with no structuretaxvaluedollarcnt value as taxvaluedollarcnt and landtaxvaluedollarcnt are also missing
    df.drop(index=62533, inplace=True)

    # drop the four rows with missing taxamount
    df.drop(index=df[df.taxamount.isna() == True].index.tolist(), inplace=True)

    # drop rows where censustractandblock is missing
    df.drop(index=df[df.censustractandblock.isna() == True].index.tolist(), inplace=True)

    # drop rows where regionidzip is missing
    df.drop(index=df[df.regionidzip.isna() == True].index.tolist(), inplace=True)

    # creating county variable
    df["county"] = df["fips"].map({6037: "Los Angeles County", 6059: "Orange County", 6111: "Ventura County"})

    # creating tax_rate variable
    df["tax_rate"] = df.taxamount / df.taxvaluedollarcnt

    # split data into train, validate, test
    train, test = train_test_split(df, train_size=.8, random_state=56, stratify=df.county)
    train, validate = train_test_split(train, train_size=.75, random_state=56, stratify=train.county)

    # KNN imputation for regionidcity
    imputer, train, validate, test = impute_regionidcity(train, validate, test)

    # scale data
    scaler, train, validate, test = scale_numeric_data(train, validate, test)

    return imputer, scaler, train, validate, test