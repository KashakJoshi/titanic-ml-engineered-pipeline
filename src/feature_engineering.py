

import pandas as pd






def extract_title(df):
    """
    Extract title from the Name column.
    Example:
    'Braund, Mr. Owen Harris' -> 'Mr'
    """
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r',\s*([^\.]+)\.')
    return df






def extract_title(df):
    df = df.copy()
    
    df["Title"] = df["Name"].str.extract(r',\s*([^\.]+)\.')
    
    # Standardize titles
    df["Title"] = df["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    })
    
    # Group rare titles
    rare_titles = ["Dr", "Rev", "Major", "Col", "Don", "Lady", "Sir", "Capt", "the Countess", "Jonkheer"]
    
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    
    return df







def create_family_features(df):
    df = df.copy()
    
    # Total family members including passenger
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    
    # IsAlone feature
    df["IsAlone"] = 0
    df.loc[df["FamilySize"] == 1, "IsAlone"] = 1
    
    # Family category
    def family_category(size):
        if size == 1:
            return "Alone"
        elif 2 <= size <= 4:
            return "Small"
        else:
            return "Large"
    
    df["FamilyCategory"] = df["FamilySize"].apply(family_category)
    
    return df







def extract_deck(df):
    df = df.copy()
    
    # Extract first letter from Cabin
    df["Deck"] = df["Cabin"].str[0]
    
    # Replace missing values with 'Unknown'
    df["Deck"] = df["Deck"].fillna("Unknown")
    
    return df






def handle_missing_values(df):
    df = df.copy()
    
    # Age → fill with median
    df["Age"] = df["Age"].fillna(df["Age"].median())
    
    # Embarked → fill with mode
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    
    # Fare → fill with median (important for test data)
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    
    return df






def drop_unused_columns(df):
    df = df.copy()
    
    columns_to_drop = ["Name", "Ticket", "Cabin"]
    
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    return df






def encode_features(df):
    df = df.copy()

    categorical_cols = [
        "Sex",
        "Embarked",
        "Title",
        "FamilyCategory",
        "Deck",
        "AgeGroup",
        "FareGroup",
        "Class_Sex"
    ]

    existing_cols = [col for col in categorical_cols if col in df.columns]

    df = pd.get_dummies(df, columns=existing_cols, drop_first=True)

    return df








def add_age_group(df):
    df = df.copy()
    
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 40, 60, 100],
        labels=["Child", "Teen", "Adult", "MidAge", "Senior"]
    )
    
    return df







def add_fare_group(df):
    df = df.copy()
    
    df["FareGroup"] = pd.qcut(
        df["Fare"],
        4,
        labels=["Low", "MidLow", "MidHigh", "High"]
    )
    
    return df







def add_class_sex_feature(df):
    df = df.copy()
    
    df["Class_Sex"] = df["Pclass"].astype(str) + "_" + df["Sex"]
    
    return df