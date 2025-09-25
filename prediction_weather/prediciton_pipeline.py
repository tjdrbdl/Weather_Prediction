# prediction_pipeline.py
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
from generate_visibility_targets import generate_target
from generate_visibility_targets import first_recovery_hour


# ========== 유틸 함수 ==========

def wc_to_group7(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    conds = [
        s.between(11,12) | (s==28) | s.between(40,49),         # Fog/Ice fog
        s.between(50,59),                                       # Drizzle
        s.between(60,69),                                       # Rain
        s.between(70,79),                                       # Solid precip
        s.between(80,89) | s.between(90,99),                    # Convective
        s.isin([4,5,6,10]) | s.between(7,9) | s.between(30,39), # Obscurations/Blowing
        s.between(0,3) | s.between(13,17) | s.between(20,29),   # Nonlocal/Recent/State
    ]
    choices = [
        "1_Fog/Ice fog","2_Drizzle","3_Rain","4_Solid precip",
        "5_Convective","6_Obscurations/Blowing","7_Nonlocal/Recent/State"
    ]
    return pd.Series(np.select(conds, choices, default="7_Nonlocal/Recent/State"), index=series.index)

# ========== 데이터 전처리 ==========

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터 전처리:
    - 결측치 처리
    - 파생 변수 생성
    - 범주형 변수 인코딩
    """
    df['TM'] = pd.to_datetime(df['TM'], format="%Y-%m-%d %H:%M:%S")
    df = df[df['TM'] >= '2016-01-02']

    # 결측치 처리
    df = df.replace(-9, np.nan)

    #파생변수 생성
    df['TA'].interpolate(method='linear', limit_direction='both', inplace=True)
    df['TD'].interpolate(method='linear', limit_direction='both', inplace=True)
    df["DPD"] = df["TA"] - df["TD"]

    # 불필요한 컬럼 제거
    columns_to_drop = ['GST', 'RVR1', 'RVR2','CH4','CA4', 'CT4','PS','PA','TA','WD','CH2','STN']
    df.drop(columns=columns_to_drop, inplace=True)

    # 결측치 처리
    df['WC'].fillna(0, inplace=True)

    df['CA1'].fillna(0, inplace=True)
    df['CT1'].fillna(0, inplace=True)
    df['CH1'].fillna(0, inplace=True)

    df['CA2'].fillna(0, inplace=True)
    df['CT2'].fillna(0, inplace=True)

    df['CA3'].fillna(0, inplace=True)
    df['CT3'].fillna(0, inplace=True)
    df['CH3'].fillna(0, inplace=True)

    df['HM'].interpolate(method='linear', limit_direction='both', inplace=True)

    df['RN'].fillna(0, inplace=True)

    # 범주형 변수 인코딩
    df["WC_grp7"] = wc_to_group7(df["WC"])
    df = pd.get_dummies(df, columns=["WC_grp7"], prefix="WC")

    #인코딩 후 원본 WC 컬럼 제거
    df = df.drop(columns=["WC"], errors="ignore")

    return df

# ========== 일 단위 피처 집계 ==========
def make_daily_features(df: pd.DataFrame, fill: str = "interpolate") -> pd.DataFrame:
    """
    일자별 24시간을 와이드 컬럼으로 인코딩.
    - 각 컬럼마다 pivot(index=date, columns=hour) → 0~23 시간 보장
    """
    d = df.copy()
    d['date'] = d['TM'].dt.date
    d['hour'] = d['TM'].dt.hour

    # 사용할 수치 컬럼 선택
    num_cols = d.select_dtypes(include=[np.number]).columns.tolist()
    drop_from_features = {"hour"}
    num_cols = [c for c in num_cols if c not in drop_from_features]

    pieces = []
    hours = list(range(24))

    for col in num_cols:
        #날짜×시간 피벗 (여분 중복 있으면 평균)
        pt = d.pivot_table(index="date", columns="hour", values=col, aggfunc="mean")

        #0~23 모든 시간 보장
        pt = pt.reindex(columns=hours)

        # reindex로 인해 생긴 NaN 값을 보간(interpolate)합니다.
        # axis=1은 행(날짜) 단위로, 즉 시간의 흐름에 따라 보간하라는 의미입니다.
        if fill == "interpolate":
            pt = pt.interpolate(axis=1, method='linear', limit_direction='both')
        elif fill is not None:
            pt = pt.fillna(fill)

        #컬럼명 태깅
        pt.columns = [f"{col}_h{h:02d}" for h in pt.columns]
        pieces.append(pt)

    daily = pd.concat(pieces, axis=1)
    daily = daily.reset_index()
    # print(daily.isna().sum())
    return daily


# ---------------------------
# 학습/검증
# ---------------------------
def train_models(X: pd.DataFrame, y_dict: Dict[str, pd.Series]) -> Dict[str, object]:
    models = {}
    for target, y in y_dict.items():
        mask = ~y.isna()
        X_t, y_t = X.loc[mask], y.loc[mask]
        
        # shuffle=False를 추가하여 데이터를 시간 순서대로 분할합니다.
        # 과거 80% 데이터로 훈련하고, 최신 20% 데이터로 검증합니다.
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_t, y_t, test_size=0.2, random_state=42, shuffle=False
        )

        # ----- 모델 후보(하이퍼파라미터 튜닝) -----
        cands = [
            RandomForestRegressor(
                n_estimators=300,     # 나무의 수를 300개로 설정
                max_depth=15,         # 나무의 최대 깊이를 15로 제한하여 과적합 방지
                min_samples_split=5,  # 노드 분할을 위한 최소 샘플 수를 5로 설정
                min_samples_leaf=3,   # 리프 노드의 최소 샘플 수를 3으로 설정
                n_jobs=-1, random_state=42
            ),
            ExtraTreesRegressor(
                n_estimators=300,     # 나무의 수를 300개로 설정
                max_depth=15,         # 나무의 최대 깊이를 15로 제한
                min_samples_split=5,  # 노드 분할을 위한 최소 샘플 수를 5로 설정
                min_samples_leaf=3,   # 리프 노드의 최소 샘플 수를 3으로 설정
                n_jobs=-1, random_state=42
            ),
            HistGradientBoostingRegressor(
                learning_rate=0.05,     # 학습률을 0.05로 설정
                max_iter=1000,          # 최대 반복 횟수를 1000으로 설정
                max_depth=7,            # 나무의 최대 깊이를 7로 제한
                l2_regularization=0.1,  # L2 규제를 약하게 적용
                random_state=42
            ),
            GradientBoostingRegressor(
                n_estimators=500,       # 나무의 수를 500개로 설정
                learning_rate=0.05,     # 학습률을 0.05로 설정
                subsample=0.8,          # 훈련 데이터의 80%만 사용하여 각 나무를 학습
                max_depth=5,            # 나무의 최대 깊이를 5로 제한
                random_state=42
            ),
            lgb.LGBMRegressor(
                n_estimators=2000,      # 조기 종료를 사용하므로 넉넉하게 설정
                learning_rate=0.02,     # 학습률을 낮게 설정하여 안정적인 학습 유도
                num_leaves=31,          # 리프 노드 수를 31개로 제한 (2^5)
                max_depth=7,            # 최대 깊이를 7로 제한
                subsample=0.8,          # 데이터 샘플링 비율
                colsample_bytree=0.8,   # 피처 샘플링 비율
                reg_alpha=0.1,          # L1 규제
                reg_lambda=0.1,         # L2 규제
                n_jobs=-1, random_state=42
            ),
            xgb.XGBRegressor(
                n_estimators=2000,      # 조기 종료를 사용하므로 넉넉하게 설정
                learning_rate=0.02,     # 학습률을 낮게 설정
                max_depth=7,            # 최대 깊이를 7로 제한
                subsample=0.8,          # 데이터 샘플링 비율
                colsample_bytree=0.8,   # 피처 샘플링 비율
                reg_alpha=0.1,          # L1 규제
                reg_lambda=0.1,         # L2 규제
                tree_method="hist", n_jobs=-1, random_state=42
            ),
            # 선형 계열(스케일링 필수)
            Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=5.0)) # 규제 강도를 5.0으로 조정
            ]),
        ]

        # ----- 학습/검증 -----
        scores = {}
        for m in cands:
            mdl = m
            try:
                if isinstance(mdl, lgb.LGBMRegressor):
                    mdl.fit(
                        X_tr, y_tr,
                        eval_set=[(X_va, y_va)],
                        eval_metric="l1",
                        callbacks=[lgb.early_stopping(100, verbose=False)]
                    )
                elif isinstance(mdl, xgb.XGBRegressor):
                    mdl.set_params(eval_metric="mae")
                    mdl.fit(
                        X_tr, y_tr,
                        eval_set=[(X_va, y_va)],
                        verbose=False,
                        early_stopping_rounds=100
                    )
                else:
                    mdl.fit(X_tr, y_tr)

                pred = mdl.predict(X_va)
                mae = mean_absolute_error(y_va, pred)
                scores[m] = mae
            except Exception as e:
                # 어떤 모델이 실패해도 전체 파이프라인은 계속
                # (예: 환경 미설치, 메모리 문제 등)
                continue
        
        # 모든 모델의 점수를 MAE가 낮은 순으로 정렬하여 출력합니다.
        print(f"----- MAE Scores for [{target}] -----")
        sorted_scores = sorted(scores.items(), key=lambda item: item[1])
        for model, score in sorted_scores:
            print(f"  - {type(model).__name__:<28}: {score:.3f}")

        best = min(scores, key=scores.get)
        print(f"==> Best for [{target}]: {type(best).__name__} (MAE={scores[best]:.3f})\n")
        models[target] = best
    return models


# ---------------------------
# 파이프라인 전체 실행
# ---------------------------
def run_pipeline(train_csv: str, test_csv: str, out_csv: str = "predictions.csv"):
    # 1) 로드 & 전처리
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    train = preprocess(train)
    test = preprocess(test)

    # 2) 일 단위 집계
    train_daily = make_daily_features(train)
    test_daily = make_daily_features(test)

    # print(train_daily.head())

    # 3) 타깃 생성 (학습 데이터에서만)
    target = generate_target(train_daily)
    target_test = generate_target(test_daily)  # 테스트셋에도 타깃 생성(평가용)
    target_test.to_csv("test_targets.csv", index=False, encoding="utf-8")

    # 4) 학습용 데이터 결합
    # 피처(train_daily)는 그대로 두고, 타깃(target)의 날짜를 하루씩 뒤로 민다.
    target['date'] = target['date'] - pd.Timedelta(days=1) 

    # 이제 D-1일의 피처와 D일의 타깃이 'date'를 기준으로 합쳐진다.
    train_df = train_daily.merge(target, on="date", how="inner")

    # 5) X, y 분리
    target_cols = ["min_visibility","recovery_1mile","recovery_3mile"]
    X = train_df.drop(columns=target_cols)
    y_dict = {c: train_df[c] for c in target_cols}

    # 6) 학습
    models = train_models(X.drop(columns=["date"]), y_dict)

    # 7) 테스트셋 예측(일자 단위)
    # D-1일의 피처로 D일을 예측하도록 학습했으므로, 테스트셋도 동일하게 맞춰줍니다.
    # test_daily에서 마지막 날 데이터를 제외하여 D-1일의 피처셋을 만듭니다.
    X_test = test_daily.iloc[:-1].drop(columns=["date"])
    
    # 예측 결과는 D일의 값이 됩니다. 따라서 날짜도 D일에 맞춰줍니다.
    pred_dates = test_daily["date"].iloc[1:]

    preds = {t: models[t].predict(X_test) for t in target_cols}
    pred_df = pd.DataFrame({"date": pred_dates, **preds})

    # 8) 예측 결과 반올림
    # min_visibility, recovery_1mile, recovery_3mile 컬럼의 값을 반올림하여 정수로 변환합니다.
    for col in target_cols:
        pred_df[col] = pred_df[col].round().astype(int)

    pred_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved -> {out_csv}")

if __name__ == "__main__":
    run_pipeline("G://Weather_Data//prediction_weather//train_validation.csv", "G://Weather_Data//prediction_weather//test.csv")