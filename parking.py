import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)  #Set GPIO number as Pin Number

# #ULTRASONIC
TRIGGER_back = 16   
ECHO_back = 19     
# 
TRIGGER_front = 12
ECHO_front =  6
# 
TRIGGER_side = 20     #The pins for the sensor on the side but currently set for back sensors for testing purposes
ECHO_side =  26

#MOTOR
in1 = 24  #GPIO24
in2 = 23  #GPIO23
en = 25   #GPIO25

maxTime = 0.04

# Direction servo
servoPIN = 21

##set GPIO direction (IN / OUT)
GPIO.setup(TRIGGER_front, GPIO.OUT)
GPIO.setup(ECHO_front, GPIO.IN)
GPIO.output(TRIGGER_front, GPIO.LOW)

GPIO.setup(TRIGGER_back, GPIO.OUT)
GPIO.setup(ECHO_back, GPIO.IN)
GPIO.output(TRIGGER_back, GPIO.LOW)

GPIO.setup(TRIGGER_side, GPIO.OUT)
GPIO.setup(ECHO_side, GPIO.IN)
GPIO.output(TRIGGER_side, GPIO.LOW)

#set GPIO direction (IN / OUT) 
GPIO.setup(servoPIN, GPIO.OUT)
steering=GPIO.PWM(servoPIN, 50)
steering.start(0)

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


time.sleep(4)


#####################################
#####   FUNCTIONS     ###############
#####################################

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

# 
def ultrasonic_back():
    GPIO.output(TRIGGER_back, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(TRIGGER_back, GPIO.LOW)

    while GPIO.input(ECHO_back)==0:
      pulse_start2 = time.time()
      
    while GPIO.input(ECHO_back)==1:
      pulse_end2 = time.time()
      
    pulse_duration2 = pulse_end2 - pulse_start2
    distance_back = pulse_duration2 * 17150
    distance_back = round(distance_back, 2)
    print ("Distance back:",distance_back,"cm")   
    return distance_back


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
     #print("ultrasonic backward") 
    
def reverse():
    # High-90, Medium-70, Low-50
    #print("forward")
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.HIGH)   
    
def stop():
    #print("ultrasonic stop")
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)   
   
def set_Speed(speed):
    throttle.ChangeDutyCycle(speed)
    
def SetAngle(angle):
    duty = angle / 18 + 1
    GPIO.output(servoPIN, True)
    steering.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(servoPIN, False)
    steering.ChangeDutyCycle(0)    
    
    
forward_speed = 38
reverse_speed = 52

right_angle = 35
left_angle = 180
straight_angle = 80

set_Speed(forward_speed)
SetAngle(180)
time.sleep(1)
SetAngle(95)
time.sleep(1)
a = 0
slot = 0
t = 0
count = 0
gap = False

parking = 0

#Finding a parking spot
while(parking == 0):
     
    distance_side = ultrasonic_side()
    
    distance_front  = ultrasonic_front()
    
    if distance_front < 20:
        stop()
        print("Object detected - Front")
    
    else:
        forward()
        if count == 0 :
            if distance_side > 45:
                gap = True
                a = time.time()
                print("asdffsfsfgsdfbsdfbsdfbsdfbx......................................")
                count = 1          
            
        if gap == True :
            if distance_side < 30:
                b = time.time()
                t = b - a
                print('--------------------------------------------------------------',t)
                gap = False            
        
        if t > 0.16:
            slot = 1
            print("Parking spot found")
            time.sleep(0.25)
            stop()
            time.sleep(1)
            parking = 1
            reverse_count = 1
        
        
# Reversing Code
set_Speed(reverse_speed)

while(reverse_count == 1):

    distance_back = ultrasonic_back()
    if distance_back > 35:
        SetAngle(180)
        print("turn")
        time.sleep(0.2)
        reverse()
        print("reversing")
        time.sleep(1.2)
        stop()
        time.sleep(0.2)
        SetAngle(40)
        time.sleep(0.2)
        forward()
        time.sleep(0.8)
        stop()
        SetAngle(170)
        time.sleep(0.2)
        reverse()
        time.sleep(1.1)
        stop()
        print("ssss")
        
        reverse_count = 2
        
    else:
        print("stop")
        stop()        
        break
        
while(reverse_count == 2):
    set_Speed(35)
    distance_back = ultrasonic_back()
    if distance_back > 20:
        SetAngle(95)
        time.sleep(0.01)
        reverse()
        print("reve")
        
    else:    
        stop()
        print("stop2")
        break
    
    
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
throttle.stop()
steering.stop()
GPIO.cleanup()
#     print("GPIO Clean up")


