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

def get_single_unit_properties(df):
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

    return df

def remove_columns(df, cols_to_remove=[]):
    """
    Docstring
    """
    df = df.drop(columns=cols_to_remove)
    return df

def handle_missing_values(df, prop_required_column = .60, prop_required_row = .60):
    """
    Docstring
    """

    # establish threshold for retaining a column
    threshold = int(round(prop_required_column*len(df.index),0))

    # drop columns where threshold criteria is not met
    df.dropna(axis=1, thresh=threshold, inplace=True)

    # establish a threshold for retaining an index
    threshold = int(round(prop_required_row*len(df.columns),0))

    # drop indices where threshold criteria is not met
    df.dropna(axis=0, thresh=threshold, inplace=True)

    return df

def impute_missing_values(df):
    """
    Docstring
    """

    # heatingorsystemdesc imputation with "None" as properties are in SoCal so having no heating system is reasonable
    df.heatingorsystemdesc = df.heatingorsystemdesc.fillna("None")

    # buildingqualitytypeid imputation
    df.buildingqualitytypeid = df.buildingqualitytypeid.fillna(df.buildingqualitytypeid.median())

    # lotsizesquarefeet imputation with mode
    df.lotsizesquarefeet = df.lotsizesquarefeet.fillna(6000.0)

    # calculatedbathnbr imputation
    df.calculatedbathnbr = df.calculatedbathnbr.fillna(df.calculatedbathnbr.median())

    # calculatedfinishedsquarefeet imputation with mode
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.fillna(1120.0)

    # fullbathcnt imputation
    df.fullbathcnt = df.fullbathcnt.fillna(df.fullbathcnt.median())

    # yearbuilt imputation
    df.yearbuilt = df.yearbuilt.fillna(round(df.yearbuilt.mean()))

    # structuretaxvaluedollarcnt imputation based on the difference between taxvaluedollarcnt and landtaxvaluedollarcnt as 
    # 99.9% of the values in structuretaxvaluedollarcnt are equal to this difference
    df.structuretaxvaluedollarcnt = df.structuretaxvaluedollarcnt.fillna(df.taxvaluedollarcnt - df.landtaxvaluedollarcnt)

    return df

def remove_rows(df, cols_to_mask=[]):
    """
    Docstring
    """

    for col in cols_to_mask:
        df.drop(index=df[df[col].isna() == True].index.tolist(), inplace=True)

    return df

def create_new_variables(df):
    """
    Function does the following:
    1. Creates the county variable using the fips codes
    2. Calculates the tax_rate variable
    3. Returns an updated DataFrame
    """

    # creating county variable
    df["county"] = df["fips"].map({6037: "Los Angeles County", 6059: "Orange County", 6111: "Ventura County"})

    # creating tax_rate variable
    df["tax_rate"] = df.taxamount / df.taxvaluedollarcnt

    return df

def get_upper_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + (k * iqr)
    return s.apply(lambda x: max([x - upper_bound, 0]))

def get_lower_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the lower outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the lower bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    lower_bound = q1 - (k * iqr)
    return s.apply(lambda x: min([x - lower_bound, 0]))

def add_upper_outlier_columns(df, k):
    '''
    Add a column with the suffix upper_outliers for all the numeric columns
    in the given dataframe.
    '''

    for col in df.select_dtypes('number'):
        df[col + '_upper_outliers'] = get_upper_outliers(df[col], k)

    return df

def add_lower_outlier_columns(df, k):
    '''
    Add a column with the suffix lower_outliers for all the numeric columns
    in the given dataframe.
    '''

    for col in df.select_dtypes('number'):
        df[col + '_lower_outliers'] = get_lower_outliers(df[col], k)

    return df

def convert_dtypes(df, columns=[], dtype="object"):
    """
    Function does the follwing:
    1. Takes in a DataFrame, specified columns to transform as a list, and the desired data type of the specified columns as a string
    2. Returns a DataFrame with columns converted to specified dtype
    """

    # convert dtype of variable
    for col in columns:
        df[col] = df[col].astype(dtype)

    return df

def prep_zillow(df):
    """
    Docstring
    """

    # get single unit properties
    df = get_single_unit_properties(df)

    # calling remove_columns
    # dropping unnecessary id columns
    # dropping propertyzoningdesc as we have already filtered the data to single unit properties
    # 99.7% of the values in finishedsquarefeet12 are the same as calculatedfinishedsquarefeet
    # drop finishedsquarefeet12 as the info appears to be redundant when compared to calculatedfinishedsquarefeet
    df = remove_columns(df, cols_to_remove=["id", "propertylandusetypeid", "heatingorsystemtypeid", "propertyzoningdesc", "finishedsquarefeet12"])

    # calling handle_missing_values to remove columns and rows that do not meet the default threshold critera necessary to retain variable or index
    df = handle_missing_values(df)

    # imputation for missing values in df
    df = impute_missing_values(df)

    # drop the rows with no structuretaxvaluedollarcnt value
    # should only drop one row where both taxvaluedollarcnt and landtaxvaluedollarcnt are missing in addition to structuretaxvaluedollarcnt since all other 
    # structuretaxvaluedollarcnt values have been imputed using the impute_structuretaxvaluedollarcnt function
    # drop rows with missing taxamount
    # drop rows where censustractandblock is missing
    # drop rows where regionidzip is missing
    df = remove_rows(df, cols_to_mask=["structuretaxvaluedollarcnt", "taxamount", "censustractandblock", "regionidzip"])

    # creating new county and tax_rate variables
    df = create_new_variables(df)

    # resetting index
    df.reset_index(inplace=True)

    # dropping former index
    df.drop(columns="index", inplace=True)

    # convert dtypes so that numeric categorical data is not captured by outlier functions
    # df = convert_dtypes(df, columns=["parcelid", "buildingqualitytypeid", "fips", "latitude", "longitude", "rawcensustractandblock", "regionidcity", "regionidcounty", "regionidzip", "censustractandblock", "logerror"], dtype="object")

    # handling upper outliers
    # df = add_upper_outlier_columns(df, 3)

    # handling lower outliers
    # df = add_lower_outlier_columns(df, 3)

    # convert back to numeric dtypes
    # df = convert_dtypes(df, columns=["parcelid"], dtype="int")
    # df = convert_dtypes(df, columns=["buildingqualitytypeid", "fips", "latitude", "longitude", "rawcensustractandblock", "regionidcity", "regionidcounty", "regionidzip", "censustractandblock", "logerror"], dtype="float")

    return df