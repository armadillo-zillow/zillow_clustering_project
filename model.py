import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import math

import scipy.stats as stats
from scipy.stats import zscore, iqr, percentileofscore, scoreatpercentile

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import sklearn.metrics
import sklearn.linear_model
from sklearn.feature_selection import RFE

import wrangle_zillow as wr
import preprocessing as pr
import explore as ex
import evaluate as ev

def create_model(df, k):
    """
    Docstring
    """

    # create features and target
    X = df.drop(columns=["parcelid", "buildingqualitytypeid", "fips", "propertycountylandusecode", "regionidcity", "regionidcounty", "regionidzip", "unitcnt", "assessmentyear", "rawcensustractandblock", "censustractandblock", "transactiondate", "heatingorsystemdesc", "propertylandusedesc", "county", "logerror"])
    y = df.logerror

    # create baseline
    df["baseline_logerror"] = df.logerror.mean()

    # creating linear model object
    lm = sklearn.linear_model.LinearRegression()

    # recursive feature elimination for k features
    rfe = RFE(lm, k)
    rfe.fit(X, y)
    X_rfe = rfe.transform(X)

    # fit models to k features predicting logerror
    lm.fit(X_rfe, y)

    # store predictions
    df['multiple_rfe'] = lm.predict(X_rfe)

    # model RMSE
    RMSE = ev.regression_errors(y, df.multiple_rfe, df)["RMSE"]
    # print(f"   Model RMSE = {RMSE}")

    # baseline RMSE
    RMSE_baseline = ev.baseline_errors(y, df.baseline_logerror, df)["RMSE_baseline"]
    # print(f"Baseline RMSE = {RMSE_baseline}")

    # evaluate
    ev.better_than_baseline(df.logerror, df.multiple_rfe, df.baseline_logerror, df)

    return df, RMSE, RMSE_baseline


