import cv2
import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

model_path = r'C:\Users\sator\Downloads\New folder (2)\mask_detection.keras'

if not os.path.exists(model_path):
    print(f'Model file not found at : {model_path}')
    exit

model = load_model(model_path)

label_dict = ['Incorrect Mask','with Mask','No Mask']
colors_dict = {0: (10,50,20), 1: (0,255,0), 2: (0, 0, 255)}


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_resize = cv2.resize(frame,(224,224))
    img_rgb = cv2.cvtColor(img_resize,cv2.COLOR_BGR2RGB)
    img_array = preprocess_input(np.expand_dims(img_rgb,axis=0))

    pred= model.predict(img_array)
    class_idx = pred[0].argmax()
    conf = pred[0][class_idx]*100
    label = f"{label_dict[class_idx]} : {conf:.2f}% "
    color = colors_dict[class_idx]

    cv2.putText(frame,label,(10,40), cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

    cv2.imshow("Mask Detector",frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break