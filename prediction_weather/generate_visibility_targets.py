import pandas as pd

def first_recovery_hour(group: pd.DataFrame, threshold: int) -> int:
    """
    시정(VS)이 threshold 이상으로 '처음' 회복된 시(hour)를 반환
    - 미회복: -1
    - group: 동일 일자 레코드 (hour, VS 필요)
    """
    s, e = -1, -1
    for time, dst in group[['hour', 'VS']].itertuples(index=False):
        if s < 0 and dst < threshold:
            s = time
        if s >= 0 and dst >= threshold:
            e = time
            break
    return e if s >= 0 and e >= 0 else -1

def generate_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    타깃 생성
      - min_visibility : 일자별 VS 최소값
      - recovery_1mile : 1마일(약 1609m) 최초 회복 시각
      - recovery_3mile : 3마일(약 4828m) 최초 회복 시각
    """
    df2 = df.copy()
    df2['date'] = df2['datetime'].dt.date
    df2['hour'] = df2['datetime'].dt.hour
    grouped = df2.groupby('date')

    # 최소 시정
    min_visibility = grouped.apply(lambda g: g['VS'].min()).rename('min_visibility')

    # 1마일(1609m) 회복시간
    recovery_1mile = grouped.apply(lambda g: first_recovery_hour(g, 1609)).rename('recovery_1mile')

    # 3마일(4828m) 회복시간
    recovery_3mile = grouped.apply(lambda g: first_recovery_hour(g, 4828)).rename('recovery_3mile')

    # 결과 병합
    target_df = pd.concat([min_visibility, recovery_1mile, recovery_3mile], axis=1).reset_index()
    return target_df