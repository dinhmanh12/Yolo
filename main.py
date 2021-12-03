import numpy as np
import cv2

cap = cv2.VideoCapture("video.mp4")

classesFile = 'coco.names'
classesNames = []

confThreshold = 0.5
nmsthreshold = 0.3 # Càng nhỏ càng chọn lọc ít hộp hơn
# Đọc các class trong file coco
with open(classesFile, 'rt') as f:
    classesNames = f.read().rstrip('\n').split('\n')

model_cfg = 'yolov4.cfg'
model_weight = 'yolov4.weights'

# Khởi tạo mạng
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weight)
# Thiết lập chạy model trên CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT , wT , cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:   # Lặp trong 3 outputs
        for det in output:   # Lặp trong ma trận 85 phần tử từng output
            scores = det[5:] # Lấy ra xác xuất của 80 class
            classId = np.argmax(scores) #Tìm giá trị lớn nhất trong ma trận 80 class
            confidence = scores[classId] # Lưu giá trị đó ở đây
            if confidence > confThreshold:
                w , h = int(det[2]*wT) ,int(det[3]*hT)  #Lấy chiều rộng và chiều cao hộp giới hạn
                center_x , center_y = int(det[0]*wT) , int(det[1]*hT) # Tọa độ góc trải hộp giới hạn
                x , y = int(center_x - w/2) , int(center_y - h/2) # Tọa độ tâm hộp giới hạn
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsthreshold) # Loại bỏ các hộp trùng lặp

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classesNames[classIds[i]].upper()}{int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
# Đọc từng frame ảnh và xử lý
while True:
    sucess, img = cap.read()
    # Xử lý ảnh đầu vào trước khi đưa vào mạng định dạng mạng hiểu được
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    # Lấy các lớp đầu ra của mạng
    layerNames = net.getLayerNames()  # Lấy tên tất cả các lớp của mạng
    outputNames = [layerNames[i[0] - 1] for i in
                   net.getUnconnectedOutLayers()]  # Lấy ra số lớp không được sử dụng (lớp cuối cùng :200,227,254)

    outputs = net.forward(outputNames) # tạo ra 3 output đầu ra tương ứng
    findObjects(outputs,img)

    cv2.imshow("IMG", img)
    cv2.waitKey(1)