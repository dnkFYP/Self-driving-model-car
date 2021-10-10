import os
from math import cos, sin, pi, floor
import pygame
from adafruit_rplidar import RPLidar

import RPi.GPIO as GPIO
import time
import cv2
import math
import sys
import numpy as np

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)  #Set GPIO number as Pin Number

TRIGGER_front = 12
ECHO_front =  6
# 
TRIGGER_side = 20     #The pins for the sensor on the side but currently set for back sensors for testing purposes
ECHO_side =  26

#MOTOR
in1 = 24  #GPIO24
in2 = 23  #GPIO23
en = 25   #GPIO25
temp1=1

# Steering servo
servoPIN = 21

#set GPIO direction (IN / OUT) 
GPIO.setup(servoPIN, GPIO.OUT)
steering=GPIO.PWM(servoPIN, 50)
steering.start(0)

##set GPIO direction (IN / OUT)
GPIO.setup(TRIGGER_front, GPIO.OUT)
GPIO.setup(ECHO_front, GPIO.IN)
GPIO.output(TRIGGER_front, GPIO.LOW)

GPIO.setup(TRIGGER_side, GPIO.OUT)
GPIO.setup(ECHO_side, GPIO.IN)
GPIO.output(TRIGGER_side, GPIO.LOW)


#set GPIO direction (IN / OUT)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(en,GPIO.OUT)
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
throttle = GPIO.PWM(en,1000)

throttle.start(82)
print ("Starting")
print("\n")


time.sleep(1)

#####################################
#####   Navigating FUNCTIONS     ###############
#####################################
maxTime = 0.04

def ultrasonic_front():
    GPIO.output(TRIGGER_front, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(TRIGGER_front, GPIO.LOW)

    while GPIO.input(ECHO_front)==0:
      pulse_start1 = time.time()
      
    while GPIO.input(ECHO_front)==1:
      pulse_end1 = time.time()
      
    pulse_duration1 = pulse_end1 - pulse_start1
    distance_front = pulse_duration1 * 17150
    distance_front = round(distance_front, 2)
    print ("Distance front:",distance_front,"cm")   
    return distance_front


def ultrasonic_side():
    GPIO.output(TRIGGER_side, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(TRIGGER_side, GPIO.LOW)
        
    pulse_start1 = time.time()
    timeout = pulse_start1 + maxTime
    while GPIO.input(ECHO_side)==0 and pulse_start1 < timeout:
      pulse_start1 = time.time()
    
    pulse_end1 = time.time()
    timeout = pulse_end1 + maxTime
    while GPIO.input(ECHO_side)==1 and pulse_end1 < timeout:
      pulse_end1 = time.time()
      
    pulse_duration1 = pulse_end1 - pulse_start1
    distance_side = pulse_duration1 * 17150
    distance_side = round(distance_side, 2)
    print ("Distance side:",distance_side,"cm")
    
    return distance_side



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
    time.sleep(1)
    GPIO.output(servoPIN, False)
    steering.ChangeDutyCycle(0)   
    
################################################################    

# Screen width & height
W = 800
H = 800

SCAN_BYTE = b'\x20'
SCAN_TYPE = 129

# Set up pygame and the display
os.putenv('SDL_FBDEV', '/dev/fb1')
pygame.display.init()
lcd = pygame.display.set_mode((W,H))
pygame.mouse.set_visible(False)
#lcd.fill((200,0,0))
pygame.display.update()

# Setup the RPLidar
PORT_NAME = '/dev/ttyUSB0'
lidar = RPLidar(None, PORT_NAME)

lidar.stop_motor()
lidar.stop()
lidar.disconnect()
lidar = RPLidar(None, PORT_NAME) 

# used to scale data to fit on the screen
max_distance = 0
turn = 0    #1 for right and 2 for left turn

#pylint: disable=redefined-outer-name,global-statement
def process_data(data):
    global max_distance
    lcd.fill((0,0,0))
    point = ( int(W / 2) , int(H / 2) )
    turn = 0
    
    pygame.draw.circle(lcd,pygame.Color(255, 255, 255),point,10 )
    pygame.draw.circle(lcd,pygame.Color(100, 100, 100),point,200 , 1 )
    pygame.draw.line( lcd,pygame.Color(100, 100, 100) , ( 0, int(H/2)),( W , int(H/2) ) )
    pygame.draw.line( lcd,pygame.Color(100, 100, 100) , ( int(W/2),0),( int(W/2) , H ) )

    for angle in range(360):
        distance = data[angle]
        if (distance < 650 and distance > 50) and (angle < 175 and angle > 140):
        #if (distance < 500 and distance > 10):
            print("Object is at angle:",angle, "so turn right")
            turn = 1
            #break
        
        if (distance < 650 and distance > 50) and (angle < 220 and angle > 185):
        #if (distance < 500 and distance > 10):
            print("Object is at angle: ",angle, "so turn left")
            turn = 1
            #break

        if distance > 0:                  # ignore initially ungathered data points
            max_distance = max([min([5000, distance]), max_distance])
            radians = (angle+90) * pi / 180.0
            x = distance * cos(radians)
            y = distance * sin(radians)
            point = ( int(W / 2) + int(x / max_distance * (W/2)), int(H/2) + int(y / max_distance * (H/2) ))
            pygame.draw.circle(lcd,pygame.Color(255, 0, 0),point,2 )
    pygame.display.update()
    return turn
   
scan_data = [0]*360

def _process_scan(raw):
    '''Processes input raw data and returns measurment data'''
    new_scan = bool(raw[0] & 0b1)
    inversed_new_scan = bool((raw[0] >> 1) & 0b1)
    quality = raw[0] >> 2
    if new_scan == inversed_new_scan:
        raise RPLidarException('New scan flags mismatch')
    check_bit = raw[1] & 0b1
       
    if check_bit != 1:
        raise RPLidarException('Check bit not equal to 1')
    angle = ((raw[1] >> 1) + (raw[2] << 7)) / 64.
    distance = (raw[3] + (raw[4] << 8)) / 4.
    return new_scan, quality, angle, distance

def lidar_measurments(self, max_buf_meas=500):
       
        lidar.set_pwm(800)
        status, error_code = self.health
        cmd = SCAN_BYTE
        self._send_cmd(cmd)
        dsize, is_single, dtype = self._read_descriptor()
        if dsize != 5:
            raise RPLidarException('Wrong info reply length')
        if is_single:
            raise RPLidarException('Not a multiple response mode')
        if dtype != SCAN_TYPE:
            raise RPLidarException('Wrong response data type')
        while True:
            raw = self._read_response(dsize)
            self.log_bytes('debug', 'Received scan response: ', raw)
            if max_buf_meas:
                data_in_buf = self._serial_port.in_waiting
                if data_in_buf > max_buf_meas*dsize:
                    self.log('warning',
                             'Too many measurments in the input buffer: %d/%d. '
                             'Clearing buffer...' %
                             (data_in_buf//dsize, max_buf_meas))
                    self._serial_port.read(data_in_buf//dsize*dsize)
            yield _process_scan(raw)

def lidar_scans(self, max_buf_meas=500, min_len=5):     #### CHANGE MIN LENGTH FOR CALIBRATION
        
        scan = []
        iterator = lidar_measurments(lidar,max_buf_meas)
        for new_scan, quality, angle, distance in iterator:
            if new_scan:
                if len(scan) > min_len:
                    yield scan
                scan = []
            if quality > 0 and distance > 0:
                scan.append((quality, angle, distance))
                
                
 #######################################################
#####    OVERTAKING       ###############
#######################################################               
def overtake_right():
    
    turn_right = 0
    
    SetAngle(right_angle)
    time.sleep(0.1)
    SetAngle(left_angle)
    time.sleep(0.35)
    SetAngle(straight_angle)
    time.sleep(0.01)
    
#     set_Speed(25)
#     time.sleep(0.001)
    
    while(turn_right == 0):
        distance_left = ultrasonic_side()
        
        if distance_left > 40:
            print("------------  TURNING LEFT  ------------")
            turn_right = 1
            
            
            SetAngle(left_angle)
            time.sleep(0.18)
#             set_Speed(forward_speed)
#             time.sleep(0.001)
            
            SetAngle(8)
            time.sleep(0.12)
            SetAngle(104)
            time.sleep(0.008)
            reverse()
            set_Speed(reverse_speed)
            time.sleep(1.4)
            
            stop()
            
        else:
            turn_right = 0
            

  
                
#######################################################
#####    MAIN CODE       ###############
#######################################################


video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH,320)
video.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

time.sleep(0.01)                
                
forward_speed = 36
reverse_speed = 52

right_angle = 10
left_angle = 180
straight_angle = 95

set_Speed(forward_speed)
SetAngle(180)
time.sleep(1)
SetAngle(95)
time.sleep(1)



try:
    distance = ultrasonic_front()

    forward()

    for scan in lidar_scans(lidar):
        for (_, angle, distance) in scan:
            scan_data[min([359, floor(angle)])] = distance
             
        turn = process_data(scan_data)
        
        if turn == 1:
            print("turning right")
            lidar.stop()
            lidar.disconnect()
            video.release()
            cv2.destroyAllWindows()
            break
        if turn == 2:
            print("turning left")
            lidar.stop()
            lidar.disconnect()
            video.release()
            cv2.destroyAllWindows()
            break
        #print(distance)
    while(turn == 1):        
        overtake_right()
        turn = 0


    
except KeyboardInterrupt:
    print('Stopping.')

throttle.stop()
steering.stop()


GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)



GPIO.cleanup()