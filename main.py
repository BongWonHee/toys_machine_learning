# conda install -c conda-forge fastapi uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#  script 오류남
# No 'Access-Control-Allow-Origin'_해결을 위해 추가 import가 필요함
# CORS 설정
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],  # 실제 운영 환경에서는 접근 가능한 도메인만 허용하는 것이 좋습니다.
    allow_methods=["*"],
    allow_headers=["*"],
)

import pickle

# /api_v1/mlmodelwithregression with dict params
# method : post
@app.post('/expectation/surgeytime') 
def mlmodelwithregression(data:dict) : # json
    print('data with dict {}'.format(data))
    # data dict to 변수 활당
    gender = float(data['gender'])
    bloodpressure = float(data['bloodpressure'])
    surgerytechnic = float(data['surgerytechnic'])
    frontdischigh = float(data['frontdischigh'])
    backdischigh = float(data['backdischigh'])
    modicchange = float(data['modicchange'])
    discwide = float(data['discwide'])


    # pkl 파일 존재 확인 코드 필요

    result_predict = 0;
    # 학습 모델 불러와 예측
    with open('datasets/test_file.plk', 'rb') as regression_file:
        loaded_model = pickle.load(regression_file)
        input_labels = [[gender,bloodpressure,surgerytechnic,frontdischigh,backdischigh,modicchange,discwide]] # 학습했던 설명변수 형식 맞게 적용
        result_predict = loaded_model.predict(input_labels)
        print('Predict radius_mean Result : {}'.format(result_predict))
        pass

    # 예측값 리턴
    result = {'surgeytime': result_predict[0]}
    return result

# test { "gender": 1, "bloodpressure": 0, "surgerytechnic": 1, "frontdischigh":10.5,"backdischigh": 37.4, "modicchange": 0, "discwide": 1248.23}