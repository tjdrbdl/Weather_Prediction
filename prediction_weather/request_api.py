import requests
import json

def download_file(file_url, save_path):
    response = requests.get(file_url)
    response.encoding = 'utf-8'  # 응답 인코딩 지정
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(response.text)  # 텍스트로 저장

# config.json 파일에서 authKey를 읽어옵니다.
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
auth_key = config['authKey']

# URL과 저장 경로 변수를 지정합니다.
url = f'https://apihub.kma.go.kr/api/typ01/url/kma_air_tm.php?tm1=201601011200&tm2=202512011200&stn=110&help=1&authKey={auth_key}'
save_file_path = 'data.txt'

# 파일 다운로드 함수를 호출합니다.
download_file(url, save_file_path)