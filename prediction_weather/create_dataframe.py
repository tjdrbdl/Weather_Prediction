import pandas as pd
import io

def create_dataframe(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text_data = f.read()

    # START 및 END 구분자 제거
    text_data = text_data.replace("#START7777\n", "").replace("#7777END\n", "")

    # 주석 제거 및 데이터 분리
    data_lines = [line.strip() for line in text_data.splitlines() if not line.startswith('#') and line.strip()]

    data = []
    for line in data_lines:
        values = line.replace(',=', '').split(',')
        data.append(values)

    # 컬럼명 정의
    columns = [
        "TM", "STN", "WD", "WS", "GST", "VS", "RVR1", "RVR2", "WC", "CA",
        "CA1", "CT1", "CH1", "CA2", "CT2", "CH2", "CA3", "CT3", "CH3", "CA4",
        "CT4", "CH4", "TA", "TD", "HM", "PA", "PS", "RN"
    ]

    # DataFrame 생성
    df = pd.DataFrame(data, columns=columns)

    # TM 열을 datetime 형식으로 변환
    df['TM'] = pd.to_datetime(df['TM'], format='%Y%m%d%H%M')

    return df

# 파일 경로
file_path = 'G:\Weather_Data\prediction_weather\data.txt'

# DataFrame 생성
weather_df = create_dataframe(file_path)

# 기준 날짜 설정
split_date = pd.to_datetime('20250901', format='%Y%m%d')

# Train/Validation과 Test 데이터 분리
train_val_df = weather_df[weather_df['TM'] < split_date]
test_df = weather_df[weather_df['TM'] >= split_date]

# Train/Validation과 Test 데이터를 각각 CSV 파일로 저장
train_val_df.to_csv('G:\\Weather_Data\\prediction_weather\\train_val_data.csv', index=False, encoding='utf-8')
test_df.to_csv('G:\\Weather_Data\\prediction_weather\\test_data.csv', index=False, encoding='utf-8')

# 결과 확인
print("Train/Validation Data:")
print(train_val_df.head())
print(train_val_df.info())

print("\nTest Data:")
print(test_df.head())
print(test_df.info())