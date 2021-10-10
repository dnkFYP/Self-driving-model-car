import RPi.GPIO as GPIO
import time
import cv2
import math
import sys
import numpy as np

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)  #Set GPIO number as Pin Number

#ULTRASONIC
TRIGGER = 12   #GPIO17
ECHO = 6      #GPIO18

#MOTOR
in1 = 24  #GPIO24
in2 = 23  #GPIO23
en = 25   #GPIO25
temp1=1

# Steering servo
servoPIN = 21

GPIO.setup(servoPIN, GPIO.OUT)
steering = GPIO.PWM(servoPIN, 50)
steering.start(60)

#set GPIO directiqon (IN / OUT)
GPIO.setup(TRIGGER, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.output(TRIGGER, GPIO.LOW)


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
    #print ("Distance:",distance,"cm")   
    return distance

def forward():
    # High-90, Medium-70, Low-50
     GPIO.output(in1,GPIO.HIGH)
     GPIO.output(in2,GPIO.LOW)
     #print("ultrasonic backward") 
    
def reverse():
    # High-90, Medium-70, Low-50
    #print("forward")
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.HIGH)   
    
    
def stop():
    print("stop")
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)   
   
def set_Speed(speed):    # High-90, Medium-70, Low-50
    throttle.ChangeDutyCycle(speed)
    
def SetAngle(angle):
    duty = angle / 18 + 1
    GPIO.output(servoPIN, True)
    steering.ChangeDutyCycle(duty)
    time.sleep(0.05)
    GPIO.output(servoPIN, False)
    steering.ChangeDutyCycle(0)      
    
    
#######################################################
#####     LANE TRACKINH FUNCTIONS       ###############
#######################################################


def detect_edges(frame):
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("HSV",hsv)
    
    ###HSV VALUES###
    lower_blue = np.array([0, 0, 0], dtype = "uint8")
    upper_blue = np.array([179, 140, 255], dtype="uint8")
    
#     #####RGB VALUES###
#     lower_blue = np.array([90, 0, 0], dtype = "uint8")
#     upper_blue = np.array([200, 95, 62], dtype="uint8")
    
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    cv2.imshow("mask",mask)
    
    # detect edges
    edges = cv2.Canny(mask, 50, 100)
    cv2.imshow("edges",edges)
    
    return edges

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus lower half of the screen
    polygon = np.array([[
        (40, height-40),
        (40, height/2 + 10 ),
        (width-10 , height/2 + 10),
        (width-10 , height-30),
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon,color =(255,255,255))
    
    cropped_edges = cv2.bitwise_and(edges, mask)
    cv2.imshow("roi",cropped_edges)
    
    return cropped_edges

def detect_line_segments(cropped_edges):
    rho = 1  
    theta = np.pi / 180  
    min_threshold = 10  
    
    line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold, 
                                    np.array([]), minLineLength=5, maxLineGap=150)

    return line_segments

def average_slope_intercept(frame, line_segments):
    lane_lines = []
    
    if line_segments is None:
        print("no line segments detected")
        return lane_lines

    height, width,_ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary
    
    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                #print("skipping vertical lines (slope = infinity")
                continue
            
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    return lane_lines

def make_points(frame, line):
    height, width, _ = frame.shape
    
    slope, intercept = line
    
    y1 = height  # bottom of the frame
    y2 = int(y1 / 2)  # make points from middle of the frame down
    
    if slope == 0:
        slope = 0.1
        
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return [[x1, y1, x2, y2]]

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6):
    line_image = np.zeros_like(frame)
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
                
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    return line_image


def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5 ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    
    steering_angle_radian = steering_angle / 180.0 * math.pi
    
    x1 = int(width/2)
    y1 = height
    x2 = int(x1 - height/2  / math.tan(steering_angle_radian))
    y2 = int(height/2)
    
    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    
    return heading_image

def get_steering_angle(frame, lane_lines):
    
    height,width,_ = frame.shape
    
    if len(lane_lines) == 2:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)
        
    elif len(lane_lines) == 1:
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)
        
    elif len(lane_lines) == 0:
        x_offset = 0
        y_offset = int(height / 2)
        
    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  
    steering_angle = angle_to_mid_deg + 90
    
    return steering_angle




#######################################################
#####    MAIN CODE       ###############
#######################################################


video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH,480)
video.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

time.sleep(0.01)

##fourcc = cv2.VideoWriter_fourcc(*'XVID')
##out = cv2.VideoWriter('Original15.avi',fourcc,10,(320,240))
##out2 = cv2.VideoWriter('Direction15.avi',fourcc,10,(320,240))

speed = 8
lastTime = 0
lastError = 0

kp = 0.4
kd = kp * 0.65

forward_speed = 38
reverse_speed = 52

right_angle = 35
left_angle = 165
straight_angle = 95

set_Speed(forward_speed)
SetAngle(180)
time.sleep(1)
SetAngle(straight_angle)
time.sleep(1)


while True:
    
    ret,frame = video.read()
    frame = cv2.flip(frame,-1)
    
    time.sleep(0.01)
    cv2.imshow("original",frame)
    edges = detect_edges(frame)
    roi = region_of_interest(edges)
    line_segments = detect_line_segments(roi)
    lane_lines = average_slope_intercept(frame,line_segments)
    lane_lines_image = display_lines(frame,lane_lines)
    steering_angle = get_steering_angle(frame, lane_lines)
    heading_image = display_heading_line(lane_lines_image,steering_angle)
    cv2.imshow("heading line",heading_image)

    now = time.time()
    dt = now - lastTime
    
    distance = ultrasonic()

    forward()
    
    if distance < 30:   #was 45
        stop()
        time.sleep(0.1)
        
    else:      
        deviation = steering_angle - 90
        #print(deviation)
        error = abs(deviation)

        if deviation < 29 and deviation > -22:     #STRAIGHT
            deviation = 0
            SetAngle(straight_angle)
            
        elif deviation > 29:                      #RIGHT
            print("right")
            SetAngle(right_angle)
            time.sleep(0.1)            

        elif deviation < -22:                  #LEFT   
            print("left")
            SetAngle(left_angle)
            time.sleep(0.1)

        derivative = kd * (error - lastError) / dt
        proportional = kp * error
        PD = int(speed + derivative + proportional)
        spd = abs(PD)

        if spd > 25:
            spd = 25

        #print("Motor running")
        forward()


        lastError = error
        lastTime = time.time()
        

    key = cv2.waitKey(1)
    if key == ord('q'):
        throttle.stop()
        steering.stop()
        break
    
video.release()
cv2.destroyAllWindows()
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
throttle.stop()
steering.stop()

GPIO.cleanup()
#     print("GPIO Clean up")