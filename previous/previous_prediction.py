# visibility_prediction.py
# -------------------------------------------
# Train:
# - T15/S2 기반 'weather_risk_score' 파생변수 생성
# - OLS p-value 기반 통계적 피처 선택 적용
# - 일자 단위 피처 집계 -> 타깃 생성(최저 시정, 16m/48m 회복시간)
# Model:
# - RandomForestRegressor 단일 모델 사용
# - 검증 데이터에 대한 성능 리포트 및 시각화 생성
# Predict:
# - 테스트 파일별 일자 집계 후 예측, result.csv 저장
# -------------------------------------------

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ========== 피처 엔지니어링 ==========

def calculate_risk_score(text: str) -> int:
    """
    T15, S2 같은 기상 현상 코드 텍스트를 기반으로 위험 점수를 계산.
    - 강도(+/-), 현상 코드(RA, SN, FG 등)에 따라 점수를 차등 부여.
    """
    if pd.isna(text): return 0
    score = 0
    tokens = text.split()
    for token in tokens:
        # 강도에 따른 기본 점수: '+'(강함)=3, '-'(약함)=1, 보통=2
        base = 3 if token.startswith('+') else (1 if token.startswith('-') else 2)
        code = token[1:] if token[0] in "+-" else token
        # 특정 기상 현상에 가중치 부여
        if code in {'RA','SN','SG','PL','GR','GS'}: base += 2  # 강수
        elif code in {'FG','BR','HZ','SA','DU'}: base += 2  # 시정 방해
        elif code in {'FC','SS','DS'}: base += 5  # 위험 기상
        score += base
    return score

def add_weather_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    'T15'와 'S2' 컬럼에 대해 `calculate_risk_score`를 적용하여
    'T15_risk', 'S2_risk', 'weather_risk_score' 파생 변수를 추가.
    """
    df['T15_risk'] = df['T15'].apply(lambda x: calculate_risk_score(str(x))) if 'T15' in df.columns else 0
    df['S2_risk'] = df['S2'].apply(lambda x: calculate_risk_score(str(x))) if 'S2' in df.columns else 0
    df['weather_risk_score'] = df['T15_risk'] + df['S2_risk']
    return df

# ========== 전처리 및 피처 선택 ==========

def drop_high_na_columns(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """결측 비율이 threshold 이상인 컬럼 드롭"""
    return df.loc[:, df.isna().mean() < threshold]

def drop_by_pvalue(X: pd.DataFrame, y: pd.Series, threshold: float = 0.05) -> pd.DataFrame:
    """
    OLS 회귀 분석의 p-value를 사용하여 통계적으로 유의미하지 않은 피처를 제거.
    - threshold보다 p-value가 낮은 피처만 선택.
    """
    model = sm.OLS(y, sm.add_constant(X)).fit()
    significant_features = model.pvalues[model.pvalues < threshold].index
    # 'const'는 상수항이므로 피처 목록에서 제외
    return X[significant_features.drop('const', errors='ignore')]

def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """
    데이터 로딩 및 기본 전처리 수행.
      - K1(datetime) 파싱, T1==0 필터링
      - 주요 연속형 변수 선형 보간
      - 범주형 변수 결측치 처리 (최빈값 또는 'None')
      - 하늘상태(T16, T19 등) 유효값 필터링 및 결측치 'SKC'로 채움
      - 결측 비율 높은 컬럼 제거
    """
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['K1'], format="%Y-%m-%d %H:%M")
    df = df[df['T1'] == 0].reset_index(drop=True)
    df.drop(columns=['K1', 'T1'], inplace=True, errors='ignore')
    # 연속형 변수 보간
    for col in ['T32','T38','T39','T40','T41','T28','T29']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
    # 범주형 변수 결측치 처리
    if 'T15' in df.columns:
        df['T15'] = df['T15'].fillna(df['T15'].mode()[0])
    if 'S2' in df.columns:
        df['S2'] = df['S2'].fillna('None')
    # 하늘상태 결측치 처리 및 유효값 필터링
    for col in ['T16','T19','T22','T25']:
        if col in df.columns:
            df[col] = df[col].fillna('SKC')
            df = df[df[col].isin(['SKC','FEW','SCT','BKN','OVC'])]
    df = drop_high_na_columns(df)
    return df

def encode_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    범주형 인코딩 및 파생 변수 추가.
      - 하늘상태(T16 등): 순서형 인코딩 (SKC < FEW < SCT < BKN < OVC)
      - 기상현상(T15, S2): 레이블 인코딩
      - `add_weather_risk_features` 호출하여 위험 점수 피처 추가
    """
    df = df.copy()
    sky_cols = [c for c in ['T16','T19','T22','T25'] if c in df.columns]
    if sky_cols:
        oe = OrdinalEncoder(categories=[['SKC','FEW','SCT','BKN','OVC']] * len(sky_cols))
        df[sky_cols] = oe.fit_transform(df[sky_cols])
    for col in ['T15','S2']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    df = add_weather_risk_features(df)
    return df

# ========== 일자 집계 & 타깃 생성 ==========

def make_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """일자 기준 수치형 컬럼 평균 집계"""
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if 'datetime' in numeric_cols:
        numeric_cols.remove('datetime')
    daily = df.groupby('date')[numeric_cols].mean().reset_index()
    return daily

def generate_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    일자별 타깃 변수 생성.
      - min_visibility: 일자별 T10(시정) 최소값
      - recovery_16m: 시정이 16m 이상으로 처음 회복된 시각 (미회복 시 -1)
      - recovery_48m: 시정이 48m 이상으로 처음 회복된 시각 (미회복 시 -1)
    """
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    min_visibility = df.groupby('date')['T10'].min().reset_index(name='min_visibility')

    def get_recovery(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
        """지정된 threshold 이상으로 시정이 회복된 첫 시각을 찾음"""
        recovery = {}
        for d, group in df.groupby('date'):
            valid = group[group['T10'] >= threshold]
            # 회복된 기록이 있으면 첫번째 시간(hour)을, 없으면 -1을 저장
            recovery[d] = valid.iloc[0]['datetime'].hour if not valid.empty else -1
        return pd.DataFrame({'date': list(recovery.keys()), f'recovery_{threshold}m': list(recovery.values())})

    recovery_16m = get_recovery(df, 16)
    recovery_48m = get_recovery(df, 48)
    # 생성된 타깃들을 'date' 기준으로 병합
    return min_visibility.merge(recovery_16m, on='date').merge(recovery_48m, on='date')

# ========== 모델 평가 및 시각화 ==========

def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, title: str):
    """실제값과 예측값을 산점도로 시각화"""
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--') # y=x 참조선
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def model_performance_report(model_dict: dict, X_val_dict: dict, y_val_dict: dict) -> pd.DataFrame:
    """
    검증 데이터셋에 대한 모델 성능(MAE)을 계산하고,
    각 타깃에 대한 예측 시각화(`plot_predictions`)를 호출.
    """
    report = {}
    for target in model_dict:
        model = model_dict[target]
        X_val = X_val_dict[target]
        y_val = y_val_dict[target]
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        report[target] = {
            'MAE': mae,
            'Samples': len(y_val)
        }
        plot_predictions(y_val, y_pred, title=f'{target} Prediction')
    return pd.DataFrame(report).T

# ========== 학습/예측 ==========

def train_models(X: pd.DataFrame, y: pd.DataFrame) -> tuple[dict, list]:
    """
    각 타깃에 대해 p-value 기반 피처 선택 후 RandomForest 모델을 학습.
    - 학습된 모델 딕셔너리와 최종 사용된 피처 목록을 반환.
    """
    models = {}
    X_val_dict, y_val_dict = {}, {}
    final_columns = []

    for col in y.columns:
        print(f"\nTraining for target: {col}")
        # p-value 기반으로 유의미한 피처 선택
        X_filtered = drop_by_pvalue(X, y[col])
        final_columns = list(X_filtered.columns)
        print(f"Selected {len(final_columns)} features based on p-value.")

        X_train, X_val, y_train, y_val = train_test_split(X_filtered, y[col], test_size=0.2, random_state=42)

        # RandomForest 모델 학습
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        models[col] = model
        X_val_dict[col] = X_val
        y_val_dict[col] = y_val

    # 검증 데이터에 대한 성능 리포트 출력
    report = model_performance_report(models, X_val_dict, y_val_dict)
    print("\nModel Performance Summary:\n", report)
    return models, final_columns

def predict_test_file(test_path: str, models: dict, expected_columns: list) -> dict:
    """
    단일 테스트 파일에 대해 전처리 및 예측 수행.
    - 학습 시 사용된 `expected_columns` 기준으로 피처를 맞춤.
    """
    test_df = load_and_preprocess(test_path)
    test_df = encode_and_clean(test_df)
    agg_test = make_daily_features(test_df)

    # 학습 시 사용된 컬럼이 테스트 데이터에 없으면 0으로 채움
    for col in expected_columns:
        if col not in agg_test.columns:
            agg_test[col] = 0
    X_test = agg_test[expected_columns]

    preds = {}
    for col, model in models.items():
        pred = model.predict(X_test)
        # 예측값은 정수로 반올림
        preds[col] = int(round(pred[0]))
    return preds

# ========== 파이프라인 ==========

def run_pipeline(train_path: str, test_paths: list[str]) -> pd.DataFrame:
    """전체 학습 및 예측 파이프라인 실행"""
    # 1. 학습 데이터 준비
    df = load_and_preprocess(train_path)
    df = encode_and_clean(df)
    df_daily = make_daily_features(df)
    target = generate_target(df)
    train_df = df_daily.merge(target, on='date')

    # 2. X, y 분리
    X = train_df.drop(columns=['date', 'min_visibility', 'recovery_16m', 'recovery_48m'], errors='ignore')
    y = train_df[['min_visibility', 'recovery_16m', 'recovery_48m']]

    # 3. 모델 학습
    models, expected_columns = train_models(X, y)

    # 4. 테스트 데이터 예측
    result = []
    for path in test_paths:
        preds = predict_test_file(path, models, expected_columns)
        result.append({'filename': os.path.basename(path).replace('.csv',''), **preds})

    # 5. 결과 저장
    result_df = pd.DataFrame(result)
    result_df.to_csv('result.csv', index=False)
    print("\nSaved -> result.csv")
    return result_df

# ========== 실행 예시 ==========
if __name__ == "__main__":
    train_path = 'train.csv'
    test_paths = sorted(glob.glob('test.csv'))
    run_pipeline(train_path, test_paths)