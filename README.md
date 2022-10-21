# 2022 종합설계
## 팀원
* 20161166 양희웅
* 20181320 송지우
***
## 주제
### 차량에게 안전한 비보호 회전 안내
* 사람이 횡단보도에 접근하면 일단 멈춘 후 상황을 파악해야 한다
* 사람이 횡단보도를 건너고 있으면 무조건 멈춰야 한다
* 이는 비보호 좌회전 시에도 동일한데, 운전자가 쉽게 확인할 수 없는 시야각이 있어서 보행자를 발견하지 못할 확률이 있다
* https://www.youtube.com/watch?v=kNRo2DryF58&t=1s
### Goal
* 횡단보도, 신호등, 보행자의 정보를 통해 차량에게 안전한 좌, 우회전 시기를 안내해주는 시스템
***
## 사용한 dataset
* 보행자, 차 (AIHub 차량 및 사람 인지 영상 데이터셋) - 26000장
    * https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=195
    * 13000장씩 나누어서 저장
        * data/car_pedestrain/dataset1
        * data/car_pedestrain/dataset2
* 횡단보도, 신호등 (SelectStar 교차로 및 화폐 정보 데이터셋 데이터셋) - 35000장
    * https://open.selectstar.ai/data-set/wesee
    * 17500장씩 나누어서 저장
        * data/cross/dataset1
        * data/cross/dataset2
* 데이터의 용량이 너무 커서 github에 업로드 하지 못했습니다
* 이후에 google drive의 링크를 공유하는 등의 방법을 통해 원활한 테스트가 진행될 수 있도록 하겠습니다
***
## 코드의 구조
* safe_turn/
    * 본 팀이 구현한 코드
    * code/
        * 데이터를 학습시키거나 분리하는 코드
        * detect/
            * 횡단보도 영상에서 차량, 보행자, 횡단보도, 신호등을 인식하여 차량에게 안전한 회전 시점을 알려줌
        * train/
            * 모델을 학습하는 코드
            * car_pedestrain/, cross/ 폴더로 나뉘어 각각 알맞는 데이터셋을 학습함
            * 주요 코드
                * car_pedestrain/YOLO_train.py
                    * 차량, 보행자 인식 모델 학습
                * cross/YOLO_train.py
                    * 횡단보도, 빨간불, 초록불 인식 모델 학습
    * data/
        * yolo를 통한 학습 시 데이터들의 정보를 알려주는 파일들
        * 각 데이터셋의 경로가 담긴 yaml형식의 파일들
