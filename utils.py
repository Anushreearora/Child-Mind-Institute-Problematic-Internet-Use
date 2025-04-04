import numpy as np
import pandas as pd
import os
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score, make_scorer, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
from sklearn.linear_model import ElasticNetCV, LassoCV, Lasso, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
#from catboost import CatBoostRegressor
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from operators import *

## Helpful Functions
def columns_start_with(df, prefix):
    return [col for col in df.columns if col.startswith(prefix)]

def columns_end_with(df, suffix):
    return [col for col in df.columns if col.endswith(suffix)]


## EDA
def EDA_1(df):
  ## Distribution of sii
  display(train.groupby('sii').min()['PCIAT-PCIAT_Total'])
  display(train.groupby('sii').max()['PCIAT-PCIAT_Total'])
  #sii_bucket = [0,31,50,80,100]

def EDA_2(df):
  ## Correlation
  covariance_matrix = train.select_dtypes(include=['float64','int64']).corr()
  display(covariance_matrix)

def EDA_3(df):
  ## Box Plot showing distribution of columns grouped by age
  vis_col = ['Physical-BMI','Physical-Height','Physical-Weight','Physical-Waist_Circumference','Physical-Diastolic_BP','Physical-HeartRate','Physical-Systolic_BP']
  for col in vis_col:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Basic_Demos-Age', y=col, data=df)
    plt.title(f'{col} by sii')
    plt.show()

def EDA_4(df):
  ## group train by age and find the mean and std
  df_grouped = df.select_dtypes(include=['float64','int64']).groupby('Basic_Demos-Age').agg(['mean', 'std'])
  return df_grouped

## Feature Engineering
def featureEngineering_1(df):
  season_cols = [col for col in df.columns if 'Season' in col]

  # Convert season columns to numerical
  for col in season_cols:
    df[col] = df[col].map({'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3})
  pred_cols = ['PCIAT-PCIAT_01','PCIAT-PCIAT_02','PCIAT-PCIAT_03','PCIAT-PCIAT_04','PCIAT-PCIAT_05','PCIAT-PCIAT_06','PCIAT-PCIAT_07','PCIAT-PCIAT_08',
               'PCIAT-PCIAT_09','PCIAT-PCIAT_10','PCIAT-PCIAT_11','PCIAT-PCIAT_12','PCIAT-PCIAT_13','PCIAT-PCIAT_14','PCIAT-PCIAT_15','PCIAT-PCIAT_16',
               'PCIAT-PCIAT_17','PCIAT-PCIAT_18','PCIAT-PCIAT_19','PCIAT-PCIAT_20','SDS-SDS_Total_Raw']
  try:
    df = df.drop(pred_cols, axis=1)
  except:
    print("Predictive Columns do not exist")
  return df

def featureEngineering_2(train,test):
  scaler = StandardScaler()
  train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
  test = pd.DataFrame(scaler.fit_transform(test), columns=test.columns)
  return train,test

## Imputation
def imputation_1(train,test):
  # Multivariate Imputation with Chain-Equation
  from sklearn.experimental import enable_iterative_imputer
  from sklearn.impute import IterativeImputer
  
  imputer = IterativeImputer(max_iter=10, random_state=0)
  mice_train = train.select_dtypes(include=['float64', 'int64'])
  mice_test = test.select_dtypes(include=['float64', 'int64'])

  mice_train = imputer.fit_transform(mice_train)
  mice_test = imputer.fit_transform(mice_test)
  mice_train_df = pd.DataFrame(mice_train, columns=train.columns)
  mice_test_df = pd.DataFrame(mice_test, columns=test.columns)

  return mice_train_df,mice_test_df

def imputation_2(train,test):
  from sklearn.impute import SimpleImputer

  # Initialize the SimpleImputer
  simple_imputer = SimpleImputer(strategy='mean')  # You can change 'mean' to 'median', 'most_frequent', or 'constant'

  # Perform SimpleImputer imputation
  simple_train = simple_imputer.fit_transform(train.select_dtypes(include=['float64', 'int64']))
  simple_test = simple_imputer.fit_transform(test.select_dtypes(include=['float64', 'int64']))
  simple_train_df = pd.DataFrame(simple_train, columns=train.columns)
  simple_test_df = pd.DataFrame(simple_test, columns=test.columns)
  
  return simple_train_df, simple_test_df

def imputation_3(train,test):
  train_stats = EDA_4(train)

  def statsImpute(row):
    # Find column that has missing value
    missing = row.isnull()
    for missing_col in row[missing].index:
      age = row['Basic_Demos-Age']
      mean = train_stats[missing_col]['mean'].loc[age]
      std = train_stats[missing_col]['std'].loc[age]
      # Fill in value with random value from normal distribution
      row[missing_col] = random.gauss(mean,std)
    return row

  stats_train_df = train.apply(statsImpute, axis=1)
  stats_train_df = stats_train_df.fillna(-1)
  stats_test_df = test.apply(statsImpute, axis=1)
  stats_test_df = stats_test_df.fillna(-1)
  return stats_train_df, stats_test_df


def imputation_4(train,test):
  from sklearn.impute import KNNImputer

  # Initialize the KNNImputer
  knn_imputer = KNNImputer(n_neighbors=5)

  # Perform KNN imputation
  knn_train = knn_imputer.fit_transform(train.select_dtypes(include=['float64', 'int64']))
  knn_test = knn_imputer.fit_transform(test.select_dtypes(include=['float64', 'int64']))
  knn_train_df = pd.DataFrame(knn_train, columns=train.columns)
  knn_test_df = pd.DataFrame(knn_test, columns=test.columns)
  return knn_train_df, knn_test_df


## TS Feature Engineering
def tsEngineering_1():
    #load_time_series("series_train.parquet").to_csv('train_ts.csv', index=False)
    #load_time_series("series_test.parquet").to_csv('test_ts.csv', index=False)
    df_train = pd.read_csv('train_ts.csv')
    df_test = pd.read_csv('test_ts.csv')

    return df_train, df_test

def tsEngineering_2(df_train, df_test):
    train_ts_encoded = perform_autoencoder(df_train.drop('id', axis=1), encoding_dim=60, epochs=100, batch_size=32)
    test_ts_encoded = perform_autoencoder(df_test.drop('id', axis=1), encoding_dim=60, epochs=100, batch_size=32)

    time_series_cols = train_ts_encoded.columns.tolist()
    train_ts_encoded["id"]=df_train["id"]
    test_ts_encoded['id']=df_test["id"]

    return train_ts_encoded, test_ts_encoded

def tsEngineering_3(num_clusters = 5, dirname = "series_train.parquet"):
    # agglomerative clustering using DTW-python
    
    def read_file(filename, dirname):
        accel =  pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet')).iloc[:5000,1:4].values
        return accel

    def compare_time_series(center, distances, dirname) -> pd.DataFrame:
        ids = os.listdir(dirname)
        a = read_file(ids[center], dirname)

        for fname in ids:
            try:
                b = read_file(fname, dirname)
                distance, path = fastdtw(a, b, dist=euclidean)
                distances[-1].append(distance)
            except:
                distances[-1].append(0)

        return ids, distances

    # Finding the reference clustering center (O(5n) = 1hr15mins)
    centers = [0]
    center = 0
    distances = []
    for i in range(num_clusters-1):
        distances.append([])
        ids, distances = compare_time_series(center, distances, dirname)
        distance_np = np.array(distances)
        distance_np *= (distance_np > distance_np.mean(axis=1)[:,None])
        for j in centers:
            distance_np[:,j] = 0
            center = np.argmax(distance_np.sum(axis=0))
        center = np.argmax(np.array(distances).sum(axis=0))
        centers.append(center)
        print(center)

    # Finding the clustering center (O(n))
    clusters = np.argmin(distances,axis=0)
    centers = list(map(int, centers))
    indexes = [x.split('=')[1] for x in ids]
    classify_train_ts = pd.DataFrame({'id':indexes, 'cluster':clusters})

    # Applying All Images to test data
    test_dirname = 'series_test.parquet'
    test_ids = os.listdir(test_dirname)
    data_list = []
    for center in centers:
        data_list.append(read_file(ids[center], dirname))
    
    clusters = []
    for fname in test_ids:
        distances = []
        for data in data_list:
            try:
                b = read_file(fname, test_dirname)
                distance, path = fastdtw(data[:-1], b, dist=euclidean)
                distances += [distance]
            except:
                distances += [1e99]
        clusters.append(np.argmin(distances))

    indexes = [x.split('=')[1] for x in test_ids]
    classify_test_ts = pd.DataFrame({'id':indexes, 'cluster':clusters})
    return classify_train_ts, classify_test_ts

def tsEngineering_4(dirname = "series_train.parquet"):
    
    def read_file(filename, dirname):
        accel =  pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet')).iloc[:5000,1:4].values
        return accel
    
    def read_reference() -> list:
        # create reference images
        data_list = []
        for file in ['1468360517106-acc-watch_tv.txt','1468367368314-acc-eat_chips.txt','1469751427337-acc-sweep.txt',
                        '1468379023854-acc-brush_teeth.txt','1471982025220-acc-wash_hands.txt',
                        '1468683416878-acc-type_on_keyboard.txt','1471982629473-acc-mop_floor.txt']:
            with open(f'user/{file}','r+') as f:
                data = f.read()
                data_list.append([list(map(float, x.split(',')[1:])) for x in data.split('\n')])
        return data_list
    
    ids = os.listdir(dirname)
    data_list = read_reference()

    clusters = []
    for fname in ids:
        distances = []
        for data in data_list:
            try:
                b = read_file(fname, dirname)
                distance, path = fastdtw(data[:-1], b, dist=euclidean)
                distances += [distance]
            except:
                distances += [1e99]
        clusters.append(np.argmin(distances))

    indexes = [x.split('=')[1] for x in ids]
    classify_ts_df = pd.DataFrame({'id':indexes, 'cluster':clusters})
    return classify_ts_df

def process_file(filename, dirname):
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop('step', axis=1, inplace=True)
    return df.describe().values.reshape(-1), filename.split('=')[1]

def read_file(filename, dirname):
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop('step', axis=1, inplace=True)
    return df.describe().values.reshape(-1), filename.split('=')[1]

def load_time_series(dirname) -> pd.DataFrame:
    ids = os.listdir(dirname)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))
    
    stats, indexes = zip(*results)
    
    df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
    df['id'] = indexes
    return df


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim*3),
            nn.ReLU(),
            nn.Linear(encoding_dim*3, encoding_dim*2),
            nn.ReLU(),
            nn.Linear(encoding_dim*2, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, input_dim*3),
            nn.ReLU(),
            nn.Linear(input_dim*3, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def perform_autoencoder(df, encoding_dim=50, epochs=50, batch_size=32):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    data_tensor = torch.FloatTensor(df_scaled)
    
    input_dim = data_tensor.shape[1]
    autoencoder = AutoEncoder(input_dim, encoding_dim)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters())
    
    for epoch in range(epochs):
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i : i + batch_size]
            optimizer.zero_grad()
            reconstructed = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}]')
                 
    with torch.no_grad():
        encoded_data = autoencoder.encoder(data_tensor).numpy()
        
    df_encoded = pd.DataFrame(encoded_data, columns=[f'Enc_{i + 1}' for i in range(encoded_data.shape[1])])
    
    return df_encoded



def apply_to_train_test(func):
    """
    Decorator to convert a function that operates on a single dataframe
    to one that operates on both train and test dataframes.
    
    Parameters:
    func (callable): A function that takes a single dataframe and returns a transformed dataframe
    
    Returns:
    callable: A function that takes train and test dataframes and returns transformed versions of both
    """
    def wrapper(train_df, test_df):
        # Apply the function to both dataframes
        train_transformed = func(train_df.copy())
        test_transformed = func(test_df.copy())
        
        # Return both transformed dataframes
        return train_transformed, test_transformed
    
    # Update the wrapper's metadata
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = f"Applies {func.__name__} to both train and test dataframes.\n\nOriginal docstring:\n{func.__doc__}"
    
    return wrapper

@apply_to_train_test
def FE_0(df): #RMSE = 0.5736768792180194
  seasons_col = columns_end_with(df, 'Season')

  # Vector Neutralization of Physical with Age and Sex
  for col in columns_start_with(df, 'Physical'):
    df[col+"_neut_ageNsex"] = regression_neut(df[col], [df['Basic_Demos-Age'],df['Basic_Demos-Sex']])

  # Deviation of Seasonality
  for season in seasons_col:
    for header in columns_start_with(df, season.replace('-Season','')):
      df[header] = regression_neut(df[header], [df[season]])

  # Winsorize to reduce outliers
  for col in df.columns:
    df[col] = winsorize(df[col])

  # Create polynomial features
  for col in df.columns:
    if col != 'sii':
      df[f'{col}^2'] = power(df[col], 2)
      df[f'{col}^3'] = power(df[col], 3)

  # Drop all seasonality columns
  for season in seasons_col:
    df = df.drop(columns=[season])
  return df

@apply_to_train_test
def FE_2(df):
  # Hypothesis that Physical affects Fitness Endurance and Fitness Endurance affects FGC
  for col1 in columns_start_with(df, 'Physical'):
    for col2 in columns_start_with(df, 'Fitness_Endurance'):
      df[col1+"_neut_"+col2] = vector_neut(df[col2], df[col1])

  for col1 in columns_start_with(df, 'Fitness_Endurance'):
    for col2 in columns_start_with(df, 'FGC'):
      df[col1+"_neut_"+col2] = vector_neut(df[col2], df[col1])

  # Vector Neutralization of Physical with Age and Sex
  for col in columns_start_with(df, 'Physical'):
    df[col+"_neut_ageNsex"] = vector_neut(df[col], [df['Basic_Demos-Age'],df['Basic_Demos-Sex']])

  # Winsorize to reduce outliers
  for col in df.columns:
    df[col+"_winsorize"] = winsorize(df[col])

  # Create polynomial features
  for col in df.columns:
    if col != 'sii':
      df[f'{col}^2'] = power(df[col], 2)
      df[f'{col}^3'] = power(df[col], 3)

  return df

@apply_to_train_test
def FE_3(df): # Regression neutralising all columns RMSE = 0.588786632439714

  seasons_col = columns_end_with(df, 'Season')
  initial_columns = set(df.columns) - set(seasons_col)

  # Create interaction terms
  for col1 in initial_columns:
    for col2 in initial_columns:
      if col1 != col2:
        df[f'{col1}_neut_{col2}'] = regression_neut(df[col1], df[col2])

  # Drop all seasonality columns
  df = df.drop(initial_columns | set(seasons_col), axis=1)

  return df

@apply_to_train_test
def FE_1(df): # Regression neutralising all columns against significant ones RMSE = 0.588786632439714

  significant_cols = ['Basic_Demos-Age', 'Basic_Demos-Sex', 'CGAS-CGAS_Score',
       'Physical-BMI', 'Physical-Height', 'Physical-Weight',
       'Physical-Waist_Circumference', 'Physical-Diastolic_BP',
       'Physical-HeartRate', 'Physical-Systolic_BP',
       'Fitness_Endurance-Max_Stage', 'Fitness_Endurance-Time_Mins',
       'Fitness_Endurance-Time_Sec', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone',
       'FGC-FGC_GSND', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD',
       'FGC-FGC_GSD_Zone', 'FGC-FGC_PU']
  seasons_col = columns_end_with(df, 'Season')
  initial_columns = set(df.columns) - set(seasons_col)

  # Create interaction terms
  for col1 in initial_columns:
    for col2 in significant_cols:
      if col1 != col2:
        df[col1] = regression_neut(df[col1], df[col2])

  for col in df.columns:
    df[col] = winsorize(df[col])

  # Create polynomial features
  for col in df.columns:
    if col != 'sii':
      df[f'{col}^2'] = power(df[col], 2)
      df[f'{col}^3'] = power(df[col], 3)

  # Drop all seasonality columns
  df = df.drop(set(seasons_col), axis=1)

  return df


def FE_02(train, test, y_train):
  '''
  Simple Autofeat model for regression and classification 
  '''
  from autofeat import AutoFeatRegressor

  afr_model = AutoFeatRegressor()
  
  train_transformed = afr_model.fit_transform(train, y_train)
  test_transformed = afr_model.transform(test)
  return train_transformed, test_transformed

def FE_03(train, test):
  """
  Featuretools for automated feature engineering using 
  Deep Feature Synthesis (DFS) to create complex features.
  
  Parameters:
  train (pd.DataFrame): Training data
  test (pd.DataFrame): Test data
  
  Returns:
  tuple: (train_feature_matrix, test_feature_matrix)
  """
  
  import featuretools as ft

  # Create an EntitySet
  es = ft.EntitySet(id="feature_engineering")

  # Add the training data as an entity
  es = es.add_dataframe(
      dataframe_name="train",
      dataframe=train.reset_index(drop=True),
      index="index",
      make_index=True
  )

  # Add the test data as an entity
  es = es.add_dataframe(
      dataframe_name="test",
      dataframe=test.reset_index(drop=True),
      index="index", 
      make_index=True
  )

  # Define transformation primitives
  trans_primitives = [
      'add_numeric', 
      'subtract_numeric', 
      'multiply_numeric', 
      'divide_numeric',
      'greater_than',
      'less_than',
      'modulo_numeric',
      'percentile',
      'cum_mean',
      'cum_sum'
  ]

  # Apply Deep Feature Synthesis to training data
  train_feature_matrix, feature_defs = ft.dfs(
      entityset=es,
      target_dataframe_name="train",  # Specify the target dataframe name explicitly
      max_depth=2,
      verbose=1,
      ignore_columns={"train": ["sii"]},  # Specify which columns to ignore in which dataframe
      trans_primitives=trans_primitives
  )

  # Apply the same feature definitions to test data
  test_feature_matrix = ft.calculate_feature_matrix(
      features=feature_defs,
      entityset=es,
      dataframes ="test"  # Specify which dataframe to use
  )

  # Remove highly correlated features
  train_feature_matrix = train_feature_matrix.fillna(0)
  test_feature_matrix = test_feature_matrix.fillna(0)

  print(f"Generated {len(train_feature_matrix.columns)} features")
  
  return train_feature_matrix, test_feature_matrix

# Code for finding optimal thresholds adapted from Michael Semenoff
# https://www.kaggle.com/code/michaelsemenoff/cmi-actigraphy-feature-engineering-selection

# Model parameters and settings
SEED = 9365
N_SPLITS = 10
VOTING = True
BASE_THRESHOLDS = [30, 50, 80]

# Model hyperparameters
LGB_PARAMS = {
    'objective': 'poisson', 
    'n_estimators': 295, 
    'max_depth': 4, 
    'learning_rate': 0.045, 
    'subsample': 0.604, 
    'colsample_bytree': 0.502, 
    'min_data_in_leaf': 100,
    'random_state': SEED,
    'verbosity': -1
}

XGB_PARAMS = {
    'objective': 'reg:tweedie', 
    'num_parallel_tree': 12, 
    'n_estimators': 236, 
    'max_depth': 3, 
    'learning_rate': 0.042, 
    'subsample': 0.716, 
    'colsample_bytree': 0.790, 
    'reg_alpha': 0.005, 
    'reg_lambda': 0.0002, 
    'tweedie_variance_power': 1.139,
    'random_state': SEED,
    'verbosity': 0
}

XGB_PARAMS_2 = {
    'objective': 'reg:tweedie', 
    'num_parallel_tree': 18, 
    'n_estimators': 175, 
    'max_depth': 3, 
    'learning_rate': 0.033, 
    'subsample': 0.616, 
    'colsample_bytree': 0.599, 
    'reg_alpha': 0.003, 
    'reg_lambda': 0.002, 
    'tweedie_variance_power': 1.171,
    'random_state': SEED,
    'verbosity': 0
}

XTREES_PARAMS = {
    'n_estimators': 500, 
    'max_depth': 15, 
    'min_samples_leaf': 20, 
    'bootstrap': False,
    'random_state': SEED
}

def round_with_thresholds(raw_preds, thresholds):
    """Convert continuous predictions to discrete categories using thresholds"""
    return np.where(raw_preds < thresholds[0], int(0),
                    np.where(raw_preds < thresholds[1], int(1),
                             np.where(raw_preds < thresholds[2], int(2), int(3))))

def optimize_thresholds(y_true, raw_preds, start_vals=None):
    """Find optimal thresholds to maximize quadratic weighted kappa"""
    if start_vals is None:
        start_vals = BASE_THRESHOLDS
        
    def fun(thresholds, y_true, raw_preds):
        rounded_preds = round_with_thresholds(raw_preds, thresholds)
        return -cohen_kappa_score(y_true, rounded_preds, weights='quadratic')

    res = minimize(fun, x0=start_vals, args=(y_true, raw_preds), method='Powell')
    assert res.success
    return res.x

def calculate_weights(series):
    """Calculate sample weights inversely proportional to class frequency"""
    bins = pd.cut(series, bins=10, labels=False)
    weights = bins.value_counts().reset_index()
    weights.columns = ['target_bins', 'count']
    weights['count'] = 1 / weights['count']
    weight_map = weights.set_index('target_bins')['count'].to_dict()
    weights = bins.map(weight_map)
    return weights / weights.mean()

def cross_validate(model, data, features, score_col, index_col, cv, sample_weights=False, verbose=False):
    """Perform cross-validation and return kappa scores and predictions"""
    kappa_scores = [] 
    oof_score_predictions = np.zeros(len(data))
    thresholds = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(data, data[index_col])):
        X_train, X_val = data[features].iloc[train_idx], data[features].iloc[val_idx]
        y_train_score = data[score_col].iloc[train_idx] 
        y_train_index = data[index_col].iloc[train_idx]
        y_val_index = data[index_col].iloc[val_idx]     
        
        # Train model with sample weights if provided
        if sample_weights:
            weights = calculate_weights(y_train_score)
            model.fit(X_train, y_train_score, sample_weight=weights)
        else:
            model.fit(X_train, y_train_score)

        y_pred_train_score = model.predict(X_train)
        y_pred_val_score = model.predict(X_val)
        
        oof_score_predictions[val_idx] = y_pred_val_score 

        # Find optimal threshold in sample 
        t_1 = optimize_thresholds(y_train_index, y_pred_train_score)
        thresholds.append(t_1)

        y_pred_val_index = round_with_thresholds(y_pred_val_score, t_1)
        kappa_score = cohen_kappa_score(y_val_index, y_pred_val_index, weights='quadratic')
        kappa_scores.append(kappa_score)
        
        if verbose:
            print(f"Fold {fold_idx}: Optimized Kappa Score = {kappa_score}")
    
    if verbose:
        print(f"## Mean CV Kappa Score: {np.mean(kappa_scores)} ##")
        print(f"## Std CV: {np.std(kappa_scores)}")
    
    return np.mean(kappa_scores), oof_score_predictions, thresholds

def ensemble_predict(train, test):
    """
    Train ensemble models and predict on test data
    
    Parameters:
    train (pd.DataFrame): Training data with 'PCIAT-PCIAT_Total' and 'sii' columns
    test (pd.DataFrame): Test data
    
    Returns:
    np.array: Final ensemble predictions
    """
    # Ensure we only use numeric columns
    train = train.select_dtypes(include=['float64', 'int64'])
    test = test.select_dtypes(include=['float64', 'int64'])
    
    # Get features (excluding target variables)
    exclude_cols = ['sii', 'PCIAT-PCIAT_Total']
    all_features = [col for col in train.columns if col not in exclude_cols]
    
    # Calculate sample weights
    weights = calculate_weights(train['PCIAT-PCIAT_Total'])
    
    # Initialize CV
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    # Initialize models
    lgb_model = LGBMRegressor(**LGB_PARAMS)
    xgb_model = XGBRegressor(**XGB_PARAMS)
    xgb_model_2 = XGBRegressor(**XGB_PARAMS_2)
    xtrees_model = ExtraTreesRegressor(**XTREES_PARAMS)
    
    print('Training LGBM model...')
    score_lgb, _, lgb_thresholds = cross_validate(
        lgb_model, train, all_features, 'PCIAT-PCIAT_Total', 'sii', kf, 
        verbose=True, sample_weights=True
    )
    lgb_model.fit(train[all_features], train['PCIAT-PCIAT_Total'], sample_weight=weights)
    test_lgb = lgb_model.predict(test[all_features])

    print('Training XGBoost model 1...')
    score_xgb, _, xgb_thresholds = cross_validate(
        xgb_model, train, all_features, 'PCIAT-PCIAT_Total', 'sii', kf, 
        verbose=True, sample_weights=True
    )
    xgb_model.fit(train[all_features], train['PCIAT-PCIAT_Total'], sample_weight=weights)
    test_xgb = xgb_model.predict(test[all_features])

    print('Training XGBoost model 2...')
    score_xgb_2, _, xgb_2_thresholds = cross_validate(
        xgb_model_2, train, all_features, 'PCIAT-PCIAT_Total', 'sii', kf, 
        verbose=True, sample_weights=True
    )
    xgb_model_2.fit(train[all_features], train['PCIAT-PCIAT_Total'], sample_weight=weights)
    test_xgb_2 = xgb_model_2.predict(test[all_features])

    print('Training ExtraTrees model...')
    score_xtrees, _, xtrees_thresholds = cross_validate(
        xtrees_model, train, all_features, 'PCIAT-PCIAT_Total', 'sii', kf, 
        verbose=True, sample_weights=True
    )
    xtrees_model.fit(train[all_features], train['PCIAT-PCIAT_Total'], sample_weight=weights)
    test_xtrees = xtrees_model.predict(test[all_features])

    # Print overall mean Kappa score for all models
    print(f'Overall Mean Kappa: {np.mean([score_lgb, score_xgb, score_xgb_2, score_xtrees])}')

    # Apply optimal thresholds
    lgb_thresholds_ens = np.mean(np.array(lgb_thresholds), axis=0)
    xgb_thresholds_ens = np.mean(np.array(xgb_thresholds), axis=0)
    xgb_2_thresholds_ens = np.mean(np.array(xgb_2_thresholds), axis=0)
    xtrees_thresholds_ens = np.mean(np.array(xtrees_thresholds), axis=0)
    
    test_lgb = round_with_thresholds(test_lgb, lgb_thresholds_ens)
    test_xgb = round_with_thresholds(test_xgb, xgb_thresholds_ens)
    test_xgb_2 = round_with_thresholds(test_xgb_2, xgb_2_thresholds_ens)
    test_xtrees = round_with_thresholds(test_xtrees, xtrees_thresholds_ens)

    # Combine predictions
    if VOTING:
        # Mode voting (majority rules)
        test_preds = np.array([test_lgb, test_xgb, test_xgb_2, test_xtrees])
        voted_test = stats.mode(test_preds, axis=0).mode.flatten().astype(int)
        final_test = voted_test
    else:
        # Weighted average
        model_weights = [score_lgb, score_xgb, score_xgb_2, score_xtrees]
        test_preds = np.array([test_lgb, test_xgb, test_xgb_2, test_xtrees])
        weighted_test = np.average(test_preds, axis=0, weights=model_weights)
        final_test = np.round(weighted_test).astype(int)
        
    return final_test