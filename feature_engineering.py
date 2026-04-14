def select_features(df):
    X = df.drop('label', axis=1)
    y = df['label']
    return X, y