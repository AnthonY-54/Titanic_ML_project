# scripts/data_preprocessing.py

import pandas as pd
import numpy as np
import pickle
import os


def load_and_preprocess():
    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")

    df = pd.concat([train_df, test_df], sort=False)
    # this df will be copied in future to keep the original data safe for future reference

    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    # converting title column - string type to int
    title_mapping = {
        'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4
    }
    df['Title'] = df['Title'].map(title_mapping)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    age_bins = [0, 16, 32, 48, 64, 100]
    df['AgeBin'] = pd.cut(df['Age'], bins=age_bins, labels=False)

    fare_bins = pd.qcut(df['Fare'], 4, retbins=True, duplicates='drop')[1]
    df['FareBin'] = pd.cut(df['Fare'], bins=fare_bins, labels=False, include_lowest=True)

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    # Save bin edges
    file_path = os.path.join(models_dir, "bin_edges.pkl")

    with open(file_path, "wb") as f:
        pickle.dump({"age_bins": age_bins, "fare_bins": fare_bins}, f)

    print("[DEBUG] Bin edges saved successfully.")

    # split back
    train_df = df[:len(train_df)]
    test_df = df[len(train_df):]

    X_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']
    X_test = test_df.copy()

    return X_train, y_train, X_test


def preprocess_test_data(df):
    df = df.copy()

    # same preprocessing as in training
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].fillna(df['Embarked'].mode()[0])
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    # converting title column - string type to int
    title_mapping = {
        'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4
    }
    df['Title'] = df['Title'].map(title_mapping)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    base_dir = os.path.dirname(os.path.dirname(__file__))
    bin_path = os.path.join(base_dir, "models", "bin_edges.pkl")

    with open(bin_path, "rb") as f:
        bins = pickle.load(f)

    age_bins = bins["age_bins"]
    fare_bins = bins["fare_bins"]

    df['AgeBin'] = pd.cut(df['Age'], bins=age_bins, labels=False)
    df['FareBin'] = pd.cut(df['Fare'], bins=fare_bins, labels=False, include_lowest=True)

    return df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title", "FamilySize", "IsAlone", "AgeBin",
               "FareBin"]]

