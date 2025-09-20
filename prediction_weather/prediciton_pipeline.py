import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from generate_visibility_targets import generate_target  # 기존 타깃 생성 함수 불러오기

# ========== 데이터 전처리 ==========

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터 전처리:
    - 결측치 처리
    - 범주형 변수 인코딩
    """
    # 결측치 처리

    
    df['WC'].fillna(0, inplace=True)  # 현재일기 코드 결측치 0으로 대체
    df['TA'].interpolate(method='linear', limit_direction='both', inplace=True)  # 기온 보간
    df['TD'].interpolate(method='linear', limit_direction='both', inplace=True)  # 이슬점온도 보간
    df['HM'].interpolate(method='linear', limit_direction='both', inplace=True)  # 상대습도 보간
    df['RN'].fillna(0, inplace=True)  # 강수량 결측치 0으로 대체

    # 시간 변수 생성
    df['date'] = pd.to_datetime(df['TM']).dt.date
    df['hour'] = pd.to_datetime(df['TM']).dt.hour

    return df

# ========== 모델 학습 ==========

def train_models(X: pd.DataFrame, y_dict: dict) -> dict:
    """
    타깃 각각에 대해 모델 학습
    - RandomForestRegressor와 GradientBoostingRegressor 중 MAE가 낮은 모델 선택
    """
    models = {}
    for target_name, y in y_dict.items():
        print(f"\nTraining for target: {target_name}")

        # 결측치 제거
        mask = ~y.isna()
        X_target = X.loc[mask]
        y_target = y.loc[mask]

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_target, y_target, test_size=0.2, random_state=42
        )

        # 모델 후보
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        gb = GradientBoostingRegressor(random_state=42)

        # 모델 학습 및 평가
        models_mae = {}
        for model in [rf, gb]:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            mae = mean_absolute_error(y_valid, y_pred)
            models_mae[model] = mae
            print(f"{type(model).__name__} MAE: {mae:.4f}")

        # 최적 모델 선택
        best_model = min(models_mae, key=models_mae.get)
        print(f"Best model for {target_name}: {type(best_model).__name__}")
        models[target_name] = best_model

    return models

# ========== 예측 ==========

def predict(models: dict, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    학습된 모델을 사용하여 테스트 데이터 예측
    """
    predictions = {}
    for target_name, model in models.items():
        predictions[target_name] = model.predict(X_test)
    return pd.DataFrame(predictions)

# ========== 파이프라인 실행 ==========

def run_pipeline(train_path: str, test_path: str):
    # 데이터 로드
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 데이터 전처리
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    # 타깃 생성 (generate_visibility_targets.py의 함수 사용)
    target_df = generate_target(train_df)

    # 학습 데이터 구성
    X_train = train_df.drop(columns=['TM', 'VS', 'date'])
    y_dict = {
        'min_visibility': target_df['min_visibility'],
        'recovery_1mile': target_df['recovery_1mile'],
        'recovery_3mile': target_df['recovery_3mile']
    }

    # 모델 학습
    models = train_models(X_train, y_dict)

    # 테스트 데이터 예측
    X_test = test_df.drop(columns=['TM', 'VS', 'date'])
    predictions = predict(models, X_test)

    # 결과 저장
    predictions.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'")

# ========== 실행 예시 ==========

if __name__ == "__main__":
    run_pipeline('train.csv', 'test.csv')