import cv2

def add_signal(src1, src2):
    rows, cols, channels = src2.shape #로고파일 픽셀값 저장
    roi = src1[150:rows+150,150:cols+150] #로고파일 필셀값을 관심영역(ROI)으로 저장함.
    
    gray = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY) #로고파일의 색상을 그레이로 변경
    ret, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY) #배경은 흰색으로, 그림을 검정색으로 변경
    mask_inv = cv2.bitwise_not(mask)
    cv2.imshow('mask',mask) #배경 흰색, 로고 검정
    cv2.imshow('mask_inv',mask_inv) # 배경 검정, 로고 흰색
    
    src1_bg = cv2.bitwise_and(roi,roi,mask=mask) #배경에서만 연산 = src1 배경 복사
    cv2.imshow('src1_bg',src1_bg)
    
    src2_fg = cv2.bitwise_and(src2,src2, mask = mask_inv) #로고에서만 연산
    cv2.imshow('src2_fg',src2_fg)
    
    dst = cv2.bitwise_or(src1_bg, src2_fg) #src1_bg와 src2_fg를 합성
    cv2.imshow('dst',dst)
    
    src1[150:rows+150,150:cols+150] = src2 #src1에 dst값 합성

    return src1

src1 = cv2.imread('./test1.png')
src2 = cv2.imread('./green.png')
ret = add_signal(src1, src2)

cv2.imshow('ret', ret)

cv2.waitKey(0)
cv2.destroyAllWindows()