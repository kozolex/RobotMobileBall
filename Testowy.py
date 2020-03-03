from gpiozero import PWMOutputDevice
#from gpiozero import DigitalOutputDevice
import time
import threading
import cv2
import argparse
import numpy as np

#tu się zaczynają definicje pinów do sterowania mostkami H

PWM_DRIVE_LEFT = 21     # ENA - H-Bridge enable pin
FORWARD_LEFT_PIN = 26   # IN1 - Forward Drive
REVERSE_LEFT_PIN = 19   # IN2 - Reverse Drive
# Motor B, Right Side GPIO CONSTANTS
PWM_DRIVE_RIGHT = 5     # ENB - H-Bridge enable pin
FORWARD_RIGHT_PIN = 6  # IN1 - Forward Drive
REVERSE_RIGHT_PIN = 13   # IN2 - Reverse Drive
 
 #ustawienia pwm na odpowiednich pinach gpio
# Initialise objects for H-Bridge GPIO PWM pins
# Set initial duty cycle to 0 and frequency to 1000
driveLeft = PWMOutputDevice(PWM_DRIVE_LEFT, True, 0, 1000)
driveRight = PWMOutputDevice(PWM_DRIVE_RIGHT, True, 0, 1000)
 
# Tutaj sterowanie kierunkiem dla silnika lewego i prawego
forwardLeft = PWMOutputDevice(FORWARD_LEFT_PIN)
reverseLeft = PWMOutputDevice(REVERSE_LEFT_PIN)
forwardRight = PWMOutputDevice(FORWARD_RIGHT_PIN)
reverseRight = PWMOutputDevice(REVERSE_RIGHT_PIN)

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', 
                help = 'path to yolo config file', default='./net/tinyv3.cfg')
ap.add_argument('-w', '--weights', 
                help = 'path to yolo pre-trained weights', default='./net/BEST.weights')
ap.add_argument('-cl', '--classes', 
                help = 'path to text file containing class names',default='./net/ping-pong.names')
args = ap.parse_args()

    

# Get names of output layers, output for YOLOv3 is ['yolo_16', 'yolo_23']
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Darw a rectangle surrounding the object and its class name 
def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    #tutaj zaczynają się definicję np skrętu w lewo,
    #jedź prosto i takie tam różne w każdym ustawia się wyjścia pinów 1/0 dla kierunki
    #oraz z jakim wypełnieniem ma to działać
def allStop():
    forwardLeft.value = False
    reverseLeft.value = False
    forwardRight.value = False
    reverseRight.value = False
    driveLeft.value = 0
    driveRight.value = 0
 
def forwardDrive(wypelnienie):
    nowe=wypelnienie/100
    forwardLeft.value = True
    reverseLeft.value = False
    forwardRight.value = True
    reverseRight.value = False
    driveLeft.value = nowe
    driveRight.value = nowe
 
def reverseDrive():
    forwardLeft.value = False
    reverseLeft.value = True
    forwardRight.value = False
    reverseRight.value = True
    driveLeft.value = 0.07
    driveRight.value = 0.07
    
def forwardTurnLeft(wypelnienie):
    forwardLeft.value = True
    reverseLeft.value = False
    forwardRight.value = False
    reverseRight.value = True
    driveLeft.value = wypelnienie
    driveRight.value = wypelnienie
 
def forwardTurnRight(wypelnienie):
    forwardLeft.value = False
    reverseLeft.value = True
    forwardRight.value = True
    reverseRight.value = False
    driveLeft.value = wypelnienie
    driveRight.value = wypelnienie

def centruj(local_obj_x, pwm_par = 100):     #funkcja centrująca obiekt na ekranie kamery
    if local_obj_x != 0: 
        pwm_signal = ((local_obj_x -160)/160) / pwm_par
        if pwm_signal > 0 :
            forwardTurnLeft(pwm_signal)
        else :
            forwardTurnRight(-pwm_signal)
        return pwm_signal
    else: allStop()

def zbieraj(local_obj_x):
    centruj(local_obj_x) # po prostu wycentruj na ekranie piłkę i jedź prosto
    forwardDrive(0.03)                       # taki sam schemat jazdy jak przy centrowaniu obiektu
    t1.start()
    
def szukaj(local_obj_x):
                               # funkcja szukania obiektu na ekranie
        forwardTurnRight(0.02)   #i tutaj się obraca sobie robot powoli i szuka piłeczki                       #obrotu robota w prawo
    #if local_obj_x != 0:         #a jeżeli coś się wydarzy czyli jakaś detekcja obiektu to
      #  zbieraj(local_obj_x) 
    # tu jest próba wykorzystania timerów do oganięcia ile poszczególna funkcja ma trwać
    #typu skręcaj w lewo przez 5s również  nie dokończone bo nie ogarnąłem

window_title= "Detector"   
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

# Load names classes
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)

#Generate color for each class randomly
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Define network from configuration file and load the weights from the given weights file
net = cv2.dnn.readNet(args.weights,args.config)

# Define video capture for default cam
cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)
cap.set(5, 90)
cap.set(12,50)

local_obj_x=0

while cv2.waitKey(1) < 0:
    
    hasframe, image = cap.read()
    #image=cv2.resize(image, (620, 480)) 
    
    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (320,320), [0,0,0], True, crop=False)
    Width = image.shape[1]
    Height = image.shape[0]
    net.setInput(blob)
    
    outs = net.forward(getOutputsNames(net))
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    for out in outs:
        local_obj_x=0 # tą zmienną wykorzystuję do wypluwania gdzie czy jest obiekt na ekranie czy go nie ma
        #print(out.shape)
        for detection in out:
            
        #each detection  has the form like this [center_x center_y width height obj_score class_1_score class_2_score ..]
            scores = detection[5:]#classes scores starts from index 5
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:            
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                local_obj_x=center_x
                
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
           
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    #dodatkowo zmiennej nadaję też wartość w pikselach gdzie piłka na obrazie jest
    
    pwm_result =centruj(local_obj_x, 25)
    #tu jest część do samego rysowania pudełek na obiektach
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        draw_pred(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
   
    # A tu wyluwa opcje w okienku takie input lag
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

    cv2.putText(image, str(pwm_result), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    
    cv2.imshow(window_title, image)
    