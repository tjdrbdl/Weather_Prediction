import pandas as pd
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from generate_visibility_targets import generate_target


# ========== 유틸 함수 ==========
def wc_to_group7(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    conds = [
        s.between(11, 12) | (s == 28) | s.between(40, 49),          # Fog/Ice fog
        s.between(50, 59),                                           # Drizzle
        s.between(60, 69),                                           # Rain
        s.between(70, 79),                                           # Solid precip
        s.between(80, 89) | s.between(90, 99),                       # Convective
        s.isin([4, 5, 6, 10]) | s.between(7, 9) | s.between(30, 39), # Obscurations/Blowing
        s.between(0, 3) | s.between(13, 17) | s.between(20, 29),     # Nonlocal/Recent/State
    ]
    choices = [
        "1_Fog/Ice fog", "2_Drizzle", "3_Rain", "4_Solid precip",
        "5_Convective", "6_Obscurations/Blowing", "7_Nonlocal/Recent/State"
    ]
    return pd.Series(np.select(conds, choices, default="7_Nonlocal/Recent/State"), index=series.index)


# ========== 데이터 전처리 ==========
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    - TM 파싱(+필요 시 날짜 필터)
    - 결측치 처리(-9 -> NaN)
    - 보간(TA/TD/HM) 후 파생(DPD)
    - 필요 컬럼 드롭, WC 7그룹 원핫
    """
    df = df.copy()
    df['TM'] = pd.to_datetime(df['TM'], format="%Y-%m-%d %H:%M:%S")
    df = df[df['TM'] >= '2016-01-02']

    # -9 → NaN
    df = df.replace(-9, np.nan)

    # 수치 캐스팅 + 보간
    for c in ('TA', 'TD', 'HM'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df[['TA', 'TD', 'HM']] = df[['TA', 'TD', 'HM']].interpolate(
        method='linear', limit_direction='both'
    )

    # 파생
    df["DPD"] = df["TA"] - df["TD"]

    # 불필요 컬럼 제거(너가 쓰던 목록 유지)
    columns_to_drop = ['GST', 'RVR1', 'RVR2', 'CH4', 'CA4', 'CT4', 'PS', 'PA', 'TA', 'WD', 'CH2', 'STN']
    df.drop(columns=[c for c in columns_to_drop if c in df.columns], inplace=True, errors="ignore")

    # 결측치 처리(층운 등 0 채움 유지)
    for c in ["WC", "CA1", "CT1", "CH1", "CA2", "CT2", "CA3", "CT3", "CH3", "RN"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # 범주형 인코딩
    if "WC" in df.columns:
        df["WC_grp7"] = wc_to_group7(df["WC"])
        df = pd.get_dummies(df, columns=["WC_grp7"], prefix="WC")
        df = df.drop(columns=["WC"], errors="ignore")

    return df


# ========== 일 단위 피처 집계(24시간 와이드) ==========
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
        pt = d.pivot_table(index="date", columns="hour", values=col, aggfunc="mean")
        pt = pt.reindex(columns=hours)

        if fill == "interpolate":
            pt = pt.interpolate(axis=1, method='linear', limit_direction='both')
        elif fill is not None:
            pt = pt.fillna(fill)

        pt.columns = [f"{col}_h{h:02d}" for h in pt.columns]
        pieces.append(pt)

    daily = pd.concat(pieces, axis=1)
    daily = daily.reset_index()
    return daily


# ---------------------------
# 학습/검증 (시간기반 홀드아웃)
# ---------------------------
def train_models(
    X: pd.DataFrame,
    y_dict: Dict[str, pd.Series],
    dates: pd.Series,
    val_ratio: float = 0.2,
    use_weight: bool = False,
) -> Dict[str, object]:
    """
    최근 일자 홀드아웃 검증:
    - 유니크 날짜 정렬 후 마지막 val_ratio 비율을 검증셋으로 사용
    """
    X_full = X.copy()
    feat_cols = [c for c in X_full.columns if c != "date"]

    # 컷오프 계산(+출력용 문자열)
    dates = pd.to_datetime(dates)
    unique_days = np.array(sorted(pd.unique(dates)))
    split_idx = max(1, int(len(unique_days) * (1 - val_ratio)))
    cutoff_day = pd.to_datetime(unique_days[split_idx])
    cutoff_str = cutoff_day.strftime('%Y-%m-%d')

    models: Dict[str, object] = {}

    for target, y in y_dict.items():
        mask = ~y.isna()
        X_t = X_full.loc[mask, feat_cols]
        y_t = y.loc[mask]
        d_t = dates.loc[mask]

        is_val = d_t >= cutoff_day
        X_tr, X_va = X_t.loc[~is_val], X_t.loc[is_val]
        y_tr, y_va = y_t.loc[~is_val], y_t.loc[is_val]

        # w_tr 변수를 루프 시작 전에 None으로 초기화합니다.
        w_tr = None
        # (선택) 중요일 가중(예: min_visibility 기준)
        if use_weight and "min_visibility" in target:
            w_tr = np.where(y_tr < 5000, 3.0, 1.0)

        # ----- 모델 후보 -----
        cands = [
            RandomForestRegressor(
                n_estimators=800, max_depth=None,
                min_samples_split=4, min_samples_leaf=2,
                max_features="sqrt", n_jobs=-1, random_state=42
            ),
            ExtraTreesRegressor(
                n_estimators=1200, max_depth=None,
                min_samples_split=4, min_samples_leaf=2,
                max_features="sqrt", n_jobs=-1, random_state=42
            ),
            HistGradientBoostingRegressor(
                learning_rate=0.06, max_iter=800,
                max_depth=None, l2_regularization=0.0,
                max_bins=255, random_state=42
            ),
            GradientBoostingRegressor(
                n_estimators=1000, learning_rate=0.03, subsample=0.7,
                max_depth=3, random_state=42
            ),
            lgb.LGBMRegressor(
                n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=-1,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.2,
                random_state=42
            ),
            xgb.XGBRegressor(
                n_estimators=5000, learning_rate=0.03,
                max_depth=6, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                tree_method="hist", n_jobs=-1, random_state=42
            ),
            Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=3.0))
            ]),
        ]

        scores = {}
        side_metrics = {}  # 회복시간용 허용오차

        for m in cands:
            mdl = m
            try:
                if isinstance(mdl, lgb.LGBMRegressor):
                    mdl.fit(
                        X_tr, y_tr,
                        sample_weight=w_tr,
                        eval_set=[(X_va, y_va)],
                        eval_metric="l1",
                        callbacks=[lgb.early_stopping(100, verbose=False)]
                    )
                elif isinstance(mdl, xgb.XGBRegressor):
                    mdl.set_params(eval_metric="mae")
                    mdl.fit(
                        X_tr, y_tr,
                        sample_weight=w_tr,
                        eval_set=[(X_va, y_va)],
                        verbose=False,
                        early_stopping_rounds=100
                    )
                else:
                    fit_kwargs = {}
                    if w_tr is not None:
                        try:
                            fit_kwargs["sample_weight"] = w_tr
                        except Exception:
                            pass
                    mdl.fit(X_tr, y_tr, **fit_kwargs)

                pred = mdl.predict(X_va)
                mae = mean_absolute_error(y_va, pred)
                scores[m] = mae

                if "recovery" in target:
                    err = np.abs(y_va - pred)
                    acc1 = (err <= 1).mean()
                    acc2 = (err <= 2).mean()
                    side_metrics[m] = (acc1, acc2)

            except Exception:
                continue

        best = min(scores, key=scores.get)
        msg = f"[{target}] Best: {type(best).__name__} | MAE={scores[best]:.3f} | cutoff_day={cutoff_str}"
        if best in side_metrics:
            acc1, acc2 = side_metrics[best]
            msg += f" | ≤1h={acc1:.2%} | ≤2h={acc2:.2%}"
        print(msg)
        models[target] = best

        # ----- 피처 중요도 시각화 -----
        # 파이프라인이 아닌 단일 모델이고, feature_importances_ 속성이 있는 경우
        if hasattr(best, 'feature_importances_'):
            importances = pd.Series(best.feature_importances_, index=X_tr.columns)
            top20 = importances.sort_values(ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x=top20.values, y=top20.index)
            plt.title(f'Feature Importance for [{target}] (Top 20)')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{target}.png')
            plt.close()

    return models


# ---------------------------
# 파이프라인 전체 실행
# ---------------------------
def run_pipeline(train_csv: str, test_csv: str, out_csv: str = "predictions.csv", val_ratio: float = 0.2):
    # 1) 로드 & 전처리(시간 해상도)
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    train = preprocess(train)
    test = preprocess(test)

    # 2) 일 × 24h 피처화
    train_daily = make_daily_features(train)
    test_daily  = make_daily_features(test)

    # 2-1) 간단한 계절 피처
    for df_ in (train_daily, test_daily):
        df_["month"] = pd.to_datetime(df_["date"]).dt.month
        df_["doy"]   = pd.to_datetime(df_["date"]).dt.dayofyear

    # 3) 타깃 생성
    target = generate_target(train_daily)

    # 4) 학습용 결합
    train_df = train_daily.merge(target, on="date", how="inner")

    # 5) X, y 분리
    target_cols = ["min_visibility", "recovery_1mile", "recovery_3mile"]
    X = train_df.drop(columns=target_cols)
    y_dict = {c: train_df[c] for c in target_cols}

    # 6) 학습(시간기반 홀드아웃)
    models = train_models(
        X, y_dict,
        dates=train_df["date"],
        val_ratio=val_ratio,
        use_weight=False  # 중요일 가중 쓰려면 True
    )

    # 7) 테스트 예측(일자 단위)
    # D-1일의 피처로 D일을 예측하도록 학습했으므로, 테스트셋도 동일하게 맞춰줍니다.
    # test_daily에서 마지막 날 데이터를 제외하여 D-1일의 피처셋을 만듭니다.
    X_test = test_daily.iloc[:-1].drop(columns=["date"])

    # 예측 결과는 D일의 값이 됩니다. 따라서 날짜도 D일에 맞춰줍니다.
    pred_dates = test_daily["date"].iloc[1:]

    # ----- 실제 테스트 타깃 생성 및 저장 -----
    test_target_df = generate_target(test_daily)
    # 예측 기간과 동일하게 날짜를 맞춰줍니다.
    actual_df = test_target_df[test_target_df['date'].isin(pred_dates)]
    actual_df.to_csv("actuals_test.csv", index=False, encoding="utf-8")

    preds = {t: models[t].predict(X_test) for t in target_cols}
    pred_df = pd.DataFrame({"date": pred_dates, **preds})

    # 8) 후처리: 반올림, 범위 제한, 그리고 -1 처리
    # min_visibility는 0 이상의 정수로 변환
    pred_df["min_visibility"] = pred_df["min_visibility"].round().clip(0).astype(int)
    
    # recovery 타깃들은 후처리로 -1과 0을 명확히 구분
    for col in ["recovery_1mile", "recovery_3mile"]:
        # 임계값(예: 0.5)보다 낮은 예측값은 모두 -1로 강제 변환
        pred_df[col] = np.where(pred_df[col] < 0.5, -1, pred_df[col])
        # 그 외의 값들은 반올림하고, 값의 범위를 -1 ~ 23 사이로 제한
        pred_df[col] = pred_df[col].round().clip(-1, 23).astype(int)

    pred_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved -> {out_csv}")

    # 9) 결과 시각화
    visualize_results(pred_df, actual_df)


# ========== 결과 시각화 함수 ==========
def visualize_results(pred_df, actual_df):
    """예측값과 실제값을 비교하는 시계열/산점도 그래프 생성"""
    # 날짜 컬럼을 datetime으로 변환
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    actual_df['date'] = pd.to_datetime(actual_df['date'])
    
    # 공통 날짜로 데이터 병합
    merged_df = pd.merge(actual_df, pred_df, on='date', suffixes=('_actual', '_pred'))
    
    for target in ["min_visibility", "recovery_1mile", "recovery_3mile"]:
        target_actual = f"{target}_actual"
        target_pred = f"{target}_pred"

        plt.figure(figsize=(15, 10))

        # 1. 시계열 그래프
        plt.subplot(2, 1, 1)
        plt.plot(merged_df['date'], merged_df[target_actual], 'o-', label='Actual')
        plt.plot(merged_df['date'], merged_df[target_pred], 'o-', label='Predicted')
        plt.title(f'Time Series: Actual vs. Predicted for [{target}]')
        plt.ylabel(target)
        plt.legend()
        plt.grid(True)

        # 2. 산점도 그래프
        plt.subplot(2, 1, 2)
        sns.regplot(x=merged_df[target_actual], y=merged_df[target_pred], line_kws={'color': 'red'})
        plt.title(f'Scatter Plot: Actual vs. Predicted for [{target}]')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'result_visualization_{target}.png')
        plt.close()
        print(f"Saved -> result_visualization_{target}.png")


if __name__ == "__main__":
    run_pipeline(
        "G://Weather_Data//prediction_weather//train_validation.csv",
        "G://Weather_Data//prediction_weather//test.csv",
        out_csv="predictions.csv",
        val_ratio=0.2  # 최근 20% 홀드아웃 (필요시 0.1로 조정)
    )
