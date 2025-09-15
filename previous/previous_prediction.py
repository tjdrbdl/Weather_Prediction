# pipeline.py
# -------------------------------------------
# Train: 일자 단위 피처 집계 -> 타깃 생성(최저 시정, 16m/48m 회복시간)
# Model: RF / GB / HistGB 중 MAE 최저 모델 채택
# Predict: 테스트 파일별 일자 집계 후 예측, result.csv 저장
# -------------------------------------------

import os
import glob
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ========== 전처리 ==========

def drop_high_na_columns(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """결측 비율이 threshold 이상인 컬럼 드롭"""
    return df.loc[:, df.isna().mean() < threshold]

def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """
    공통 로딩 + 1차 정리
      - K1(datetime) 파싱
      - T1==0 필터링 후 T1 삭제
      - 일부 연속형 보간, 범주형 기본값 보정
      - 결측 심한 컬럼 삭제, 행 결측 드롭
    """
    df = pd.read_csv(filepath)

    # 시간
    df['datetime'] = pd.to_datetime(df['K1'], format='%Y-%m-%d %H:%M')
    df.drop(columns=['K1'], inplace=True, errors='ignore')

    # 관측 이상치 플래그 제거 후 필터링
    if 'T1' in df.columns:
        df = df[df['T1'] == 0].reset_index(drop=True)
        df.drop(columns=['T1'], inplace=True, errors='ignore')

    # 연속형 센서 보간(좌/우 보간 모두 허용)
    for col in ['T28', 'T29', 'T32', 'T38', 'T39', 'T40', 'T41']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')

    # 범주형 결측 처리
    if 'T15' in df.columns:
        df['T15'] = df['T15'].fillna(df['T15'].mode().iloc[0]) if not df['T15'].mode().empty else df['T15'].fillna('None')
    if 'S2' in df.columns:
        df['S2'] = df['S2'].fillna('None')

    # 하늘상태 기본값 채움
    for col in ['T16', 'T19', 'T22', 'T25']:
        if col in df.columns:
            df[col] = df[col].fillna('SKC')

    # 하늘상태에 허용된 값만 남기기(있으면)
    sky_set = {'SKC', 'FEW', 'SCT', 'BKN', 'OVC'}
    for col in ['T16', 'T19', 'T22', 'T25']:
        if col in df.columns:
            df[col] = df[col].where(df[col].isin(sky_set), 'SKC')

    # 결측 심한 컬럼 제거 + 남은 결측 행 제거
    df = drop_high_na_columns(df, threshold=0.9)
    df = df.dropna()

    return df

def encode_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    범주형 인코딩
      - T16/T19/T22/T25 : 순서형(구름량) -> SKC<FEW<SCT<BKN<OVC
      - T15, S2 : 간단 LabelEncoding (훈련/추론에서 동일하게 fit_transform 사용한 최종 흐름을 반영)
    """
    df = df.copy()

    # 구름량 순서형 인코딩
    sky_cols = [c for c in ['T16','T19','T22','T25'] if c in df.columns]
    if sky_cols:
        # 4개 칼럼 모두 같은 카테고리 순서 사용
        oe = OrdinalEncoder(categories=[['SKC', 'FEW', 'SCT', 'BKN', 'OVC']] * len(sky_cols))
        df[sky_cols] = oe.fit_transform(df[sky_cols])

    # 간단 라벨 인코딩(최종 스샷대로 fit_transform 유지)
    if 'T15' in df.columns:
        le_t15 = LabelEncoder()
        df['T15'] = le_t15.fit_transform(df['T15'])
    if 'S2' in df.columns:
        le_s2 = LabelEncoder()
        df['S2'] = le_s2.fit_transform(df['S2'])

    return df

# ========== 일자 집계 & 타깃 생성 ==========

def make_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    일자 기준 수치형 컬럼 평균
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if 'datetime' in numeric_cols:
        numeric_cols.remove('datetime')

    daily = df.groupby('date')[numeric_cols].mean().reset_index()
    return daily

def first_recovery_hour(group: pd.DataFrame, threshold: int) -> int:
    """
    시정(T10)이 threshold 이상으로 '처음' 회복된 시(hour)를 반환
    - 미회복: -1
    - group: 동일 일자 레코드 (hour, T10 필요)
    """
    s, e = -1, -1
    for time, dst in group[['hour', 'T10']].itertuples(index=False):
        if s < 0 and dst < threshold:
            s = time
        if s >= 0 and dst >= threshold:
            e = time
            break
    return e if s >= 0 and e >= 0 else -1

def generate_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    타깃 생성
      - min_visibility : 일자별 T10 최소값
      - recovery_16m   : 16m 최초 회복 시각
      - recovery_48m   : 48m 최초 회복 시각
    """
    df2 = df.copy()
    df2['date'] = df2['datetime'].dt.date
    df2['hour'] = df2['datetime'].dt.hour
    grouped = df2.groupby('date')

    min_visibility = grouped.apply(lambda g: g['T10'].min()).rename('min_visibility')
    recovery_16m  = grouped.apply(lambda g: first_recovery_hour(g, 16)).rename('recovery_16m')
    recovery_48m  = grouped.apply(lambda g: first_recovery_hour(g, 48)).rename('recovery_48m')

    target_df = pd.concat([min_visibility, recovery_16m, recovery_48m], axis=1).reset_index()
    return target_df

# ========== 학습/예측 ==========

def train_models(X: pd.DataFrame, y_dict: dict):
    """
    타깃 각각에 대해 3개 모델(RF/GB/HistGB) 중 MAE가 가장 낮은 모델을 선택.
    반환: models = { target_name: (best_model, expected_columns) }
    """
    models = {}
    model_candidates = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42)
    }

    for target_name, y in y_dict.items():
        print(f"\nTraining for target: {target_name}")

        # y가 있는 행으로 제한 + X 컬럼 결측 제거(열 기준)
        mask = ~y.isna()
        X_target = X.loc[mask]
        y_target = y.loc[mask]
        # 열 단위 결측 드랍(훈련 안정성)
        X_target = X_target.dropna(axis=1)

        expected_columns = X_target.columns.tolist()
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_target.fillna(0), y_target, test_size=0.2, random_state=42
        )

        best_model = None
        best_mae = float('inf')

        for name, model in model_candidates.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            mae = mean_absolute_error(y_valid, y_pred)
            print(f">>> {name}  MAE: {mae:.4f}")
            if mae < best_mae:
                best_mae = mae
                best_model = model

        print(f">>> Best model for {target_name}: {type(best_model).__name__} (MAE: {best_mae:.4f})")
        models[target_name] = (best_model, expected_columns)

    return models

def predict_test_file(path: str, models: dict) -> dict:
    """
    단일 테스트 파일 예측.
    - 학습 시 사용한 expected_columns 기준으로 X_test 서브셋팅
    - 각 타깃 예측값 반환
    """
    df_test = load_and_preprocess(path)
    df_test = encode_columns(df_test)
    daily = make_daily_features(df_test)

    pred_result = {'filename': os.path.basename(path).replace('.csv', '')}

    for target, packed in models.items():
        model, expected_cols = packed
        # 없는 컬럼은 0으로 보충
        for col in expected_cols:
            if col not in daily.columns:
                daily[col] = 0
        X_test = daily[expected_cols].fillna(0)

        # 마지막 1일만 활용하고 싶다면 아래 주석 해제
        # X_test = X_test.tail(1)

        # 일 단위 평균이므로 1일 1행이 일반적. 1행 기준으로 예측.
        pred_val = float(model.predict(X_test.iloc[[0]])[0])
        # 정수로 반올림 (시정분/시각 단위의 정수화)
        pred_int = int(round(pred_val))

        # 회복시간의 유효범위 제약(0~23), 아니면 -1
        if target.startswith('recovery') and not (0 <= pred_int <= 23):
            pred_int = -1

        pred_result[target] = pred_int

    # 추가 제약: 회복시간은 해당 임계값 이상 회복 필요
    # (min_visibility >= 16 / 48 인 경우, 회복시간은 -1)
    mv = pred_result.get('min_visibility', None)
    if mv is not None:
        if 'recovery_16m' in pred_result and mv >= 16:
            pred_result['recovery_16m'] = -1
        if 'recovery_48m' in pred_result and mv >= 48:
            pred_result['recovery_48m'] = -1

    return pred_result

# ========== 파이프라인 ==========

def run_pipeline(train_path: str, test_paths: list[str]) -> pd.DataFrame:
    # Train 준비
    df = load_and_preprocess(train_path)
    df = encode_columns(df)
    daily_features = make_daily_features(df)
    target_df = generate_target(df)

    # 일자 병합
    train_df = daily_features.merge(target_df, on='date', how='inner')

    # X / y 구성
    X = train_df.drop(columns=['date', 'min_visibility', 'recovery_16m', 'recovery_48m'], errors='ignore')
    y_dict = {
        'min_visibility': train_df['min_visibility'],
        'recovery_16m':  train_df['recovery_16m'],
        'recovery_48m':  train_df['recovery_48m']
    }

    # 모델 학습
    models = train_models(X, y_dict)

    # 테스트 예측
    rows = []
    for path in test_paths:
        preds = predict_test_file(path, models)
        rows.append(preds)

    result_df = pd.DataFrame(rows)
    result_df.to_csv('result.csv', index=False)
    print('Saved -> result.csv')
    return result_df


# ========== 실행 예시 ==========
if __name__ == "__main__":
    train_path = 'train.csv'
    test_paths = sorted(glob.glob('test.csv'))
    run_pipeline(train_path, test_paths)
