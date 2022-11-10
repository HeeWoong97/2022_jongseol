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
        * **prediction 관련 주요코드**
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
        * **안전범위 설정 관련 코드**
            ``` python
            safe_x1, safe_y1 = 0, 0
            safe_x2, safe_y2 = 0, 0

            def click_event(event, x, y, flags, param):
                global cnt, isClick, isFinish
                global safe_x1, safe_y1, safe_x2, safe_y2
                if isFinish:
                    return

                if isClick is False:
                    if cnt == 1:
                        print('Click the right down position')
                    elif cnt == 2:
                        print('Click the upper position')
                    elif cnt == 3:
                        print('Finish... Please press the \'0\'')
                        isFinish = True
                        return
                    isClick = True
                
                if event == cv2.EVENT_LBUTTONDOWN:
                    print(x, ' ', y)
                    if cnt == 0:
                        safe_x1 = x
                    elif cnt == 1:
                        safe_x2 = x
                    elif cnt == 2:
                        safe_y1, safe_y2 = y, y
                    cnt += 1
                    isClick = False
            ```
        * **안전범위를 통한 메인 알고리즘**
            ```python
            # safety 체크 알고리즘
            in_safety, in_cross = False, False

            if len(peds):
                for ped in peds:
                px1, py1, px2, py2 = ped

                _in_safety = int(safe_y1) <= int(py2) <= int(cy1) and int(safe_x1) <= int(px1) and int(px2) <= int(safe_x2)
                _in_cross = int(cy1) <= int(py2) <= int(cx2) and int(cx1) <= int(px1) and int(px2) <= int(cx2)
                in_safety, in_cross = in_safety or _in_safety, in_cross or _in_cross

                # red : stop!; yellow : stop and go; green : drive slowly
                if in_cross:
                result_img[start_row:start_row+rows, start_col:start_col+cols] = red
                elif in_safety:
                if cross_light_color == None:
                    result_img[start_row:start_row+rows, start_col:start_col+cols] = red
                elif cross_light_color == '초록불':
                    result_img[start_row:start_row+rows, start_col:start_col+cols] = red
                else:
                    result_img[start_row:start_row+rows, start_col:start_col+cols] = yellow
                else:   
                if cross_light_color == None:
                    result_img[start_row:start_row+rows, start_col:start_col+cols] = yellow
                elif cross_light_color == '초록불':
                    result_img[start_row:start_row+rows, start_col:start_col+cols] = yellow
                else:
                    result_img[start_row:start_row+rows, start_col:start_col+cols] = green
            else:
                # no ped
                result_img[start_row:start_row+rows, start_col:start_col+cols] = green
            ```
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
            * detect_show_split.py
                * 입력 영상에서 횡단보도 인식을 통해 보행자, 차량에게 안전 정보를 알려주는 코드
        * train/
            * 모델을 학습하는 코드
            * car_pedestrain/, cross/ 폴더로 나뉘어 각각 알맞는 데이터셋을 학습함
            * 주요 코드
                * car_pedestrain/
                    * 차량, 보행자 인식 모델 관련
                    * YOLO_train.py
                        * yolo 모델을 이용하여 모델을 학습하는 코드
                    * convert.py
                        * AI-HUB에서 제공한 label파일을 yolo 형식으로 변환하는 코드
                    * convert_train.py
                        * train set의 label을 yolo 형식으로 변환하는 코드
                    * convert_valid.py
                        * valid set의 label을 yolo 형식으로 변환하는 코드
                    * draw_box.py
                        * 변환이 올바르게 되었는지 확인하기 위해 변환된 label 형식으로 사진상에 box를 그려보는 코드
                * cross/
                    * 횡단보도, 빨간불, 초록불 인식 모델 관련
                    * YOLO_train.py
                        * yolo 모델을 이용하여 모델을 학습하는 코드
                    * convert_class.py
                        * 차량, 보행자 데이터와 횡단보도, 신호등 모델을 합칠 때 클래스 번호를 변환하는 코드
                    * convert_train.py
                        * train set의 label을 yolo 형식으로 변환하는 코드
                    * convert_valid.py
                        * valid set의 label을 yolo 형식으로 변환하는 코드
                    * preprocess.py
                        * dataset을 train, valid로 나누고 label을 yolo 형식으로 변환하는 코드
                    * split.py
                        * dataset을 train, valid로 나누는 코드
    * data/
        * yolo를 통한 학습 시 데이터들의 정보를 알려주는 파일들
        * 각 데이터셋의 경로가 담긴 yaml형식의 파일들
***
## 실행 방법
* 모델 학습 방법
    * 차량, 보행자 모델 학습
        ```
        $ python safe_turn/code/train/car_pedestrain/YOLO_train.py
        ```
    * 횡단보도, 신호등 모델 학습
        ```
        $ python safe_turn/code/train/cross/YOLO_train.py
        ```
* 모델 실행 방법
    1. 코드 실행
        ```
        $ python yolov5/detect_local.py
        ```
    2. 신호등 존재 여부 입력
        ```
        [Check presence of traffic light]
        Is there traffic light? [y/n]
        ```
        존재할 경우 y, 존재하지 않을 경우 n
    3. 안전범위 설정
        ```
        [Check the pedestrain safety range]
        Click the left down position
        351   407
        Click the right down position
        1049   401
        Click the upper position
        541   345
        Finish... Please press any key...
        safe_x1, safe_y1, safe_x2, safe_y2 =  351 345 1049 345
        ```
        안전범위의 왼쪽 아래 클릭
        안전범위의 오른쪽 아래 클릭
        안전범위의 위쪽 부분 클릭
        키를 입력하여 안전범위 설정 종료
    4. 모델이 실행된다
        ```
        [Run the model]
        31%|█████████████████████████████████████████                                                                                            | 738/2390 [01:51<04:06,  6.70it/s]
        ```
***
## 실행 결과
1. 보행자가 없어서 우회전이 가능한 경우에는 초록불로 표시한다.
<img width="1440" alt="스크린샷 2022-11-09 오후 2 57 01" src="https://user-images.githubusercontent.com/53477646/200751368-0fd2105c-0d43-4e0f-99e3-e6845b38076e.png">

2. 보행자가 안전범위에 있어서 주의가 요한 경우에는 노란불로 표시한다.
<img width="1440" alt="스크린샷 2022-11-09 오후 2 57 16" src="https://user-images.githubusercontent.com/53477646/200751663-2fd4b7f7-f741-4b28-aeaf-14200abc055b.png">

3. 보행자가 횡단보도를 건너고 있어 정지가 필요한 경우에는 빨간불로 표시한다.
<img width="1440" alt="스크린샷 2022-11-09 오후 3 03 02" src="https://user-images.githubusercontent.com/53477646/200751988-acb124fa-5e7a-4105-8d00-19e7bfe317a1.png">