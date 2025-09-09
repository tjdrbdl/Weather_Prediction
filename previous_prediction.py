#본선 task2 원본코드
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt

def calculate_risk_score(text):
    if pd.isna(text): return 0
    score = 0
    tokens = text.split()
    for token in tokens:
        base = 3 if token.startswith('+') else (1 if token.startswith('-') else 2)
        code = token[1:] if token[0] in "+-" else token
        if code in {'RA','SN','SG','PL','GR','GS'}: base += 2
        elif code in {'FG','BR','HZ','SA','DU'}: base += 2
        elif code in {'FC','SS','DS'}: base += 5
        score += base
    return score

def add_weather_risk_features(df):
    df['T15_risk'] = df['T15'].apply(lambda x: calculate_risk_score(str(x))) if 'T15' in df.columns else 0
    df['S2_risk'] = df['S2'].apply(lambda x: calculate_risk_score(str(x))) if 'S2' in df.columns else 0
    df['weather_risk_score'] = df['T15_risk'] + df['S2_risk']
    return df

def drop_high_na_columns(df, threshold=0.9):
    return df.loc[:, df.isna().mean() < threshold]

def drop_by_pvalue(X, y, threshold=0.05):
    model = sm.OLS(y, sm.add_constant(X)).fit()
    return X[model.pvalues[model.pvalues < threshold].index.drop('const', errors='ignore')]

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['K1'], format="%Y-%m-%d %H:%M")
    df = df[df['T1'] == 0].reset_index(drop=True)
    df.drop(columns=['K1', 'T1'], inplace=True)
    for col in ['T32','T38','T39','T40','T41','T28','T29']:
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
    df['T15'] = df['T15'].fillna(df['T15'].mode()[0])
    df['S2'] = df['S2'].fillna('None')
    for col in ['T16','T19','T22','T25']:
        df[col] = df[col].fillna('SKC')
        df = df[df[col].isin(['SKC','FEW','SCT','BKN','OVC'])]
    df = drop_high_na_columns(df)
    return df

def encode_and_clean(df):
    oe = OrdinalEncoder(categories=[['SKC','FEW','SCT','BKN','OVC']])
    for col in ['T16','T19','T22','T25']:
        df[col] = oe.fit_transform(df[[col]])
    for col in ['T15','S2']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    df = add_weather_risk_features(df)
    return df

def make_daily_features(df):
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if 'datetime' in numeric_cols:
        numeric_cols.remove('datetime')
    daily = df.groupby('date')[numeric_cols].mean().reset_index()
    return daily

def generate_target(df):
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    min_visibility = df.groupby('date')['T10'].min().reset_index(name='min_visibility')
    def get_recovery(df, threshold):
        recovery = {}
        for d, group in df.groupby('date'):
            valid = group[group['T10'] >= threshold]
            recovery[d] = valid.iloc[0]['datetime'].hour if not valid.empty else -1
        return pd.DataFrame({'date': list(recovery.keys()), f'recovery_{threshold//16}m': list(recovery.values())})
    recovery_16m = get_recovery(df, 16)
    recovery_48m = get_recovery(df, 48)
    return min_visibility.merge(recovery_16m, on='date').merge(recovery_48m, on='date')

def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def model_performance_report(model_dict, X_val_dict, y_val_dict):
    report = {}
    for target in model_dict:
        model = model_dict[target]
        X_val = X_val_dict[target]
        y_val = y_val_dict[target]
        y_pred = model.predict(X_val)
        mae = np.mean(np.abs(y_val - y_pred))
        report[target] = {
            'MAE': mae,
            'Samples': len(y_val)
        }
        plot_predictions(y_val, y_pred, title=f'{target} Prediction')
    return pd.DataFrame(report).T

def train_models(X, y):
    models = {}
    X_val_dict, y_val_dict = {}, {}
    for col in y.columns:
        X_filtered = drop_by_pvalue(X, y[col])
        X_train, X_val, y_train, y_val = train_test_split(X_filtered, y[col], test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        models[col] = model
        X_val_dict[col] = X_val
        y_val_dict[col] = y_val
    report = model_performance_report(models, X_val_dict, y_val_dict)
    print("\nModel Performance Summary:\n", report)
    return models, list(X_filtered.columns)

def predict_test_file(test_path, models, expected_columns):
    test_df = load_and_preprocess(test_path)
    test_df = encode_and_clean(test_df)
    agg_test = make_daily_features(test_df)
    for col in expected_columns:
        if col not in agg_test.columns:
            agg_test[col] = 0
    X_test = agg_test[expected_columns]
    preds = {}
    for col in models:
        pred = models[col].predict(X_test)
        preds[col] = int(round(pred[0]))
    return preds

def run_pipeline(train_path, test_paths):
    df = load_and_preprocess(train_path)
    df = encode_and_clean(df)
    df_daily = make_daily_features(df)
    target = generate_target(df)
    train_df = df_daily.merge(target, on='date')
    X = train_df.drop(columns=['date', 'min_visibility', 'recovery_16m', 'recovery_48m'])
    y = train_df[['min_visibility', 'recovery_16m', 'recovery_48m']]
    models, expected_columns = train_models(X, y)
    result = []
    for path in test_paths:
        preds = predict_test_file(path, models, expected_columns)
        result.append({'filename': os.path.basename(path), **preds})
    result_df = pd.DataFrame(result)
    result_df.to_csv('result.csv', index=False)
    return result_df
