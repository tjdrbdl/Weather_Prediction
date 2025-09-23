import pandas as pd
import numpy as np

def first_recovery_hour(row: pd.Series, threshold: int) -> int:
    """
    와이드 포맷의 단일 행(Series)에서 저시정 회복 시간을 찾습니다.
    - row: 'VS_h00', 'VS_h01', ... 컬럼을 포함하는 날짜별 데이터 행
    - threshold: 회복 기준으로 삼을 시정 값 (m)
    - 반환값: 회복된 시각(hour), 23시까지 미회복 시 24, 저시정 발생 없으면 -1
    """
    # 0시부터 23시까지의 시정 컬럼 이름 리스트 생성
    vis_cols = [f"VS_h{h:02d}" for h in range(24)]
    
    # 행에 시정 컬럼이 없는 경우 처리
    if not all(c in row.index for c in vis_cols):
        return -1

    # 해당 날짜의 시간대별 시정 값만 추출
    vis_values = row[vis_cols]

    # 시정이 한 번이라도 threshold 미만으로 떨어진 적이 있는지 확인
    if not (vis_values < threshold).any():
        return -1  # 저시정 이벤트가 없었으므로 -1 반환

    # 시정이 가장 낮았던 시간(hour) 찾기
    # idxmin()은 최솟값의 '인덱스'(여기서는 컬럼 이름)를 반환합니다. (예: 'VS_h05')
    min_vis_col = vis_values.idxmin()
    start_hour = int(min_vis_col.split('_h')[-1])

    # 최저 시정 시간 이후부터 순회하며 회복 시점 찾기
    for h in range(start_hour, 24):
        current_vis_col = f"VS_h{h:02d}"
        if row[current_vis_col] >= threshold:
            return h  # 회복 시점의 시간(hour) 반환

    return 24  # 23시까지 회복되지 않으면 24 반환

def generate_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    '와이드' 포맷의 일별 집계 데이터에서 타깃 변수를 생성합니다.
      - min_visibility : 일자별 VS 최소값
      - recovery_1mile : 1마일(약 1609m) 최초 회복 시각
      - recovery_3mile : 3마일(약 4828m) 최초 회복 시각
    """
    df2 = df.copy()
    
    # 0시부터 23시까지의 시정 컬럼 이름 리스트 생성
    vis_cols = [f"VS_h{h:02d}" for h in range(24)]

    # 일별 최저 시정 계산 (행 단위로 최솟값 계산)
    df2['min_visibility'] = df2[vis_cols].min(axis=1)

    # 저시정(1마일, 3마일) 발생 후 최초 회복 시간 계산
    df2["recovery_1mile"] = df2.apply(
        lambda row: first_recovery_hour(row, threshold=1609), axis=1
    )
    df2["recovery_3mile"] = df2.apply(
        lambda row: first_recovery_hour(row, threshold=4828), axis=1
    )

    # 필요한 타깃 컬럼과 date만 선택하여 반환
    target_cols = ["date", "min_visibility", "recovery_1mile", "recovery_3mile"]
    return df2[target_cols]