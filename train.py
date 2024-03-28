from ultralytics import YOLO
import cv2
import os

def train():
    model = YOLO(r'D:\박해성\rune\runs\detect\train9\weights\last.pt')
    # results = model.train(data = r'D:\박해성\rune\rune_detect.v3-2064image.yolov8\data.yaml', epochs=300, batch=2, device=0)
    results = model.train(resume=True)

def test(path):
    file_list = os.listdir(path)
    img_list = []
    for i in file_list:
        if i[-1] != 'g':
            pass
        image = cv2.imread(path + '\\' + i, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h,w,_ = image.shape
        image = image[:,int(w/2-h/2):int(w/2+h/2)]
        # cv2.imshow('image',image)
        # cv2.waitKey(0)
        img_list.append(image)

    #pt 파일 경로 입력
    model = YOLO(r'D:\박해성\rune\runs\detect\train9\weights\best.pt')

    #save=True 추가 시 인식 된 이미지 저장
    results = model(img_list,save=True, conf=0.5)

    #x좌표 순으로 정렬
    # boxes = results[0].boxes
    # arrow = [(box.xyxy.tolist(), box.conf.tolist(), box.cls.tolist()) for box in boxes]
    # arrow.sort(key = lambda x: x[0][0][0])

    #x좌표 순으로 방향키 출력 0,1,2,3 -> 하 좌 우 상
    # print([(a[2],a[1]) for a in arrow])

if __name__ =='__main__':
    # train()
    test(r'D:\박해성\rune\test_image')
