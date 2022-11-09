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
* yolov5/
    * 메인 코드가 위치
    * **detect_local.py**
        * 메인 알고리즘 코드
        * **prediction 관련 코드**
            * 이미지 프로세싱 관련
            ``` python
            # input image processing
            def img_process(img, stride, device):
                img_input = letterbox(img, img_size, stride = stride)[0]
                img_input = img_input.transpose((2, 0, 1))[::-1]
                img_input = np.ascontiguousarray(img_input)
                img_input = torch.from_numpy(img_input).to(device)
                img_input = img_input.float()
                img_input /= 255.
                img_input = img_input.unsqueeze(0)

                return img_input
            ```
            * class 검출 관련
            ``` python
            # predict classes
            def pred_classes(pred, class_names:list, ignore_class_names:list, annotator, colors)->dict:
                assert class_names == ped_class_names or class_names == cross_class_names, 'given class names are not allowed'

                preds = {class_name:[] for class_name in class_names if class_name not in ignore_class_names}

                for p in pred:
                    class_name = class_names[int(p[5])]
                    # x1, y1, x2, y2
                    position = p[:4]

                    if class_name not in ignore_class_names:
                        preds[class_name].append(position)
                    if annotator is not None:
                        annotator.box_label(position, '%s %d' % (class_name, float(p[4]) * 100), color=colors[int(p[5])])

                return preds
            ```
            * model에 이미지를 넣고 인식
            ``` python
            def detect(img, stride, device, model, class_names, ignore_class_names, colors, annotator=None):
                global cx1, cy1, cx2, cy2

                img_input = img_process(img, stride, device)

                pred = model(img_input, augment = False, visualize = False)[0]
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det = max_det)[0]
                pred = pred.cpu().numpy()

                pred[:, :4] = scale_coords(img_input.shape[2:], pred[:, :4], img.shape).round()
                preds = pred_classes(pred, class_names, ignore_class_names, annotator, colors)

                return preds
            ```
            * 설명설명설명
        * 
* safe_turn/
    * 기타 시도한 코드들
    * code/
        * 데이터를 학습시키거나 분리하는 코드
        * detect/
            * 횡단보도 영상에서 차량, 보행자, 횡단보도, 신호등을 인식하여 차량에게 안전한 회전 시점을 알려줌
            * cross_detect.py
                * 영상처리를 통해 입력 영상에서 횡단보도를 찾아보는 코드(1)
                * HSV, houghlinep등의 방법을 사용
            * cross_test.py
                * 영상처리를 통해 입력 영상에서 횡단보도를 찾아보는 코드(2)
                * 이미지 화질 변경, find contour등의 함수 사용
            * detect.py
                * 입력 영상에서 횡단보도 인식을 통해 보행자, 차량에게 안전 정보를 알려주는 코드
                * 인식 모델로는 횡단보도, 차량 인식 모델을 각각 사용한다
                * 핵심 코드
            * detect_show_split.py
                * 입력 영상에서 횡단보도 인식을 통해 보행자, 차량에게 안전 정보를 알려주는 코드
        * train/
            * 모델을 학습하는 코드
            * car_pedestrain/, cross/ 폴더로 나뉘어 각각 알맞는 데이터셋을 학습함
            * 주요 코드
                * car_pedestrain/
                    * 차량, 보행자 인식 모델 관련
                    * YOLO_train.py
                        * haha
                    * convert.py
                        * hoho
                    * convert_train.py
                        * huhu
                    * convert_valid.py
                        * hihi
                    * draw_box.py
                        * hyhy
                * cross/
                    * 횡단보도, 빨간불, 초록불 인식 모델 관련
                    * YOLO_train.py
                        * haha
                    * convert_class.py
                        * hoho
                    * convert_train.py
                        * huhu
                    * convert_valid.py
                        * hihi
                    * preprocess.py
                        * hwhw
                    * split.py
                        * jwjw
    * data/
        * yolo를 통한 학습 시 데이터들의 정보를 알려주는 파일들
        * 각 데이터셋의 경로가 담긴 yaml형식의 파일들
***
## 실행 결과
