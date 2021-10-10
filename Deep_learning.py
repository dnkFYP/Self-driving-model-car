import RPi.GPIO as GPIO
import time
import cv2
import math
import sys
import numpy as np

import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)  #Set GPIO number as Pin Number

#ULTRASONIC
TRIGGER = 12   #GPIO17
ECHO = 6      #GPIO18

#set GPIO directiqon (IN / OUT)
GPIO.setup(TRIGGER, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.output(TRIGGER, GPIO.LOW)

#MOTOR
in1 = 24  #GPIO24
in2 = 23  #GPIO23
en = 25   #GPIO25
temp1=1

# Steering servo
servoPIN = 21

GPIO.setup(servoPIN, GPIO.OUT)
steering = GPIO.PWM(servoPIN, 50)

GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(en,GPIO.OUT)
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)

throttle = GPIO.PWM(en,1000)     # set the switching frequency to 1000 Hz
throttle.stop()
throttle.start(80)

print ("Code is running...")
print("\n")


time.sleep(1)


#####################################
#####   Navigating FUNCTIONS     ###############
#####################################
def stopmovement():
    print("stop")
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)   

def forward():
    # High-90, Medium-70, Low-50
     GPIO.output(in1,GPIO.HIGH)
     GPIO.output(in2,GPIO.LOW)
     print("forward") 
    
def reverse():
    # High-90, Medium-70, Low-50
    print("reverse")
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.HIGH)
  
   
def set_Speed(speed):    # High-90, Medium-70, Low-50
    throttle.ChangeDutyCycle(speed)
    
def SetAngle(angle):
    duty = angle / 18 + 1
    GPIO.output(servoPIN, True)
    steering.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(servoPIN, False)
    steering.ChangeDutyCycle(0)     
    
def stop():
    print("stop")
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)
    
def ultrasonic():
    GPIO.output(TRIGGER, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(TRIGGER, GPIO.LOW)

    while GPIO.input(ECHO)==0:
      pulse_start = time.time()
      
    while GPIO.input(ECHO)==1:
      pulse_end = time.time()
      
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    print ("Distance:",distance,"cm")   
    return distance
#######################################################
####         Deep stuff     #############
#######################################################

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=500):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.60)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 55      #127.5
input_std = 55        #127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=500).start()               #####changed framerate from 30 to 60 for entire code
time.sleep(0.02)


#######################################################
#####    MAIN CODE       ###############
#######################################################

#speed = 50
# set_Speed(30)
# SetAngle(80)
# time.sleep(0.5)

forward_speed = 30
reverse_speed = 52

right_angle = 10
left_angle = 180
straight_angle = 95

set_Speed(forward_speed)
# SetAngle(180)
# time.sleep(0.5)
# SetAngle(75)
# time.sleep(2)

s_counter = 1
p_counter = 1
while True:
    
    distance = ultrasonic()

    forward()
    
    if distance < 30:   #was 45
        stopmovement()
    
    else:
        
        forward()
        set_Speed(forward_speed)
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()
        frame1 = cv2.flip(frame1,-1)
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        t2 = cv2.getTickCount()
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
        
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                            
                
                if object_name != "???" :
                
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    # Draw label
                    
                    
                    
                    
                if object_name == "stop sign" :
                    print("Stop sign detected.")
                    
                    if s_counter == 1:
                        set_Speed(0)
                        time.sleep(4)
                    s_counter = s_counter+1
                    
                    
                if object_name == "person" :
                    print("Person detected.")
                    if p_counter > 3  and p_counter<7:
                        set_Speed(0)
                        time.sleep(4)
                    p_counter = p_counter+1
    #                 else:
    #                     print("Person is too close")
    #                     set_Speed(0)
    #                     #time.sleep(2)
                 
                if object_name != "person" and object_name != "stop sign" :
                    set_Speed(30)
                    print("--------------    NO OBJ  --------------")
                
                
    # Draw framerate in corner of frame
  #   cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(60,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.LOW)
        throttle.stop()
        steering.stop()
        break
    


GPIO.cleanup()
#     print("GPIO Clean up")
