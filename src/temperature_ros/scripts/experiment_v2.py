#!/usr/bin/env python3

import rospy
import numpy as np
import copy
from enum import Enum, auto

from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Temperature
from std_msgs.msg import Empty


from temperature_ros import RobotController as RC
from temperature_ros import utils




# =============================================
# Config class to set experiment parameters
# =============================================  
class Config():
    
    auto = True # Enable auto switch across state. Otherwise use: rostopic pub /activate std_msgs/Empty "{}" -1  

    # remove detected object outside interesting area
    area_x_min = 340
    area_x_max = 413
    area_y_min = 190
    area_y_max = 337
    # classes name of object that must be considered
    desired_classes = ['bottle','cup']

    # hard coded trajectoy
    approaching_trajectory_names = ['rest','approach','grasp']
    release_cold_trajectory_names = ['hover','cold_release']
    release_warm_trajectory_names = ['hover','warm_release']
    release_hot_trajectory_names = ['hover','hot_release']

    # grasp closure percentage 
    sensing_closure = 0.52
    grasping_closure = 0.6

    # time of sensing [s] 
    sensing_duration = 20

    # classification bound [Volt] 
    #sensing_cold = [2,1.2]
    #sensing_warm = [1.2,1]
    #sensing_hot = [1,0]

    # path to config files
    classes_path = f'src/yolov7-ros/config/class_labels/coco.txt'
    #control_points_path = f'src/temperature_ros/config/test.yaml'
    control_points_path = f'src/temperature_ros/config/exp2.yaml'

def detect_class(output):
    if output>1.5:
        output_class = "metal" 
    elif output>1.1 and output<1.5:
        output_class = "plastic" 
    elif output<1.1:
        output_class = "wood" 
    return output_class

def set_destination(output_class, config):
    if output_class == 'metal':
        release_dest = config.release_cold_trajectory_names
    elif output_class == 'plastic':
        release_dest = config.release_warm_trajectory_names
    elif output_class == 'wood':
        release_dest = config.release_hot_trajectory_names
    else:
        release_dest = config.release_warm_trajectory_names
    return release_dest

class Status(Enum):
    IDLE = auto()                # do nothing
    OBJECT_DETECTION = auto()    # detect object with vision
    APPROACH = auto()            # approaching to sense position
    GRASP = auto()               # grasp object
    SENSE = auto()               # detect temperature with sensor
    MOVE = auto()                # move grasped object
# =============================================================================
# Callback
# =============================================================================  
object_msg = None
"""
_Detection2DArray_
Header header
# A list of the detected proposals. A multi-proposal detector might generate
#   this list with many candidate detections generated from a single input.
Detection2D[] detections
# Defines a 2D detection result.
#
# This is similar to a 2D classification, but includes position information,
#   allowing a classification result for a specific crop or image point to
#   to be located in the larger image.

_Detection2D_
Header header
# Class probabilities
ObjectHypothesisWithPose[] results
# 2D bounding box surrounding the object.
BoundingBox2D bbox
# The 2D data that generated these results (i.e. region proposal cropped out of
#   the image). Not required for all use cases, so it may be empty.
sensor_msgs/Image source_img

_ObjectHypothesisWithPose_
# An object hypothesis that contains position information.
# The unique numeric ID of object detected. To get additional information about
#   this ID, such as its human-readable name, listeners should perform a lookup
#   in a metadata database. See vision_msgs/VisionInfo.msg for more detail.
int64 id
# The probability or confidence value of the detected object. By convention,
#   this value should lie in the range [0-1].
float64 score
# The 6D pose of the object hypothesis. This pose should be
#   defined as the pose of some fixed reference point on the object, such a
#   the geometric center of the bounding box or the center of mass of the
#   object.
# Note that this pose is not stamped; frame information can be defined by
#   parent messages.
# Also note that different classes predicted for the same input data may have
#   different predicted 6D poses.
geometry_msgs/PoseWithCovariance pose

_BoundingBox2D_
# A 2D bounding box that can be rotated about its center.
# All dimensions are in pixels, but represented using floating-point
#   values to allow sub-pixel precision. If an exact pixel crop is required
#   for a rotated bounding box, it can be calculated using Bresenham's line
#   algorithm.
# The 2D position (in pixels) and orientation of the bounding box center.
geometry_msgs/Pose2D center
# The size (in pixels) of the bounding box surrounding the object relative
#   to the pose of its center.
float64 size_x
float64 size_y
"""
def camera_callback(msg:Detection2DArray):
    global object_msg
    object_msg = copy.deepcopy(msg)



temperature_msgs = None
"""
_Temperature_
# Single temperature reading.
Header header           # timestamp is the time the temperature was measured
                        # frame_id is the location of the temperature reading
float64 temperature     # Measurement of the Temperature in Degrees Celsius
float64 variance        # 0 is interpreted as variance unknown
"""
def t_callback(msg:Temperature):
    global temperature_msgs
    if temperature_msgs is not None: temperature_msgs.append(msg)


#status, transition
def deactivate_callback(msg:Empty):
    global status, transition
    print("Triggering Deactivation")
    status = Status.IDLE
    transition = False
def activate_callback(msg:Empty):
    global status, transition
    #print("Triggering Transition")
    if status==Status.OBJECT_DETECTION:
        print("")
        status=Status.APPROACH
    transition = True


# =============================================================================
# Main
# =============================================================================  
def main():
    global object_msg, temperature_msgs, status, transition

    rospy.init_node("temperature_experiment", anonymous=True)
    rate = rospy.Rate(1)  # Loop rate [Hz]
    config = Config()
    # State Machine variables
    status = Status.IDLE
    transition = True
    
    robot_controller = RC.RobotController(control_points = config.control_points_path)
    #callback triggers 
    camera_sub = rospy.Subscriber("/yolov7", Detection2DArray, camera_callback)
    temperature_sub = rospy.Subscriber("/temperature", Temperature, t_callback)
    deactivate_sub = rospy.Subscriber("/deactivate", Empty, deactivate_callback)
    activate_sub = rospy.Subscriber("/activate", Empty, activate_callback)

    classes = utils.parse_classes_file(config.classes_path)
    #utils.print_ascii_art("CIMaINa TEMPERATURE EXPERIMENT")
    while not rospy.is_shutdown():   
        rate.sleep()
        if status == Status.IDLE:
            # ======================================
            #   IDLE
            # ======================================
            if transition:
                robot_controller.back_to_ready() # <---------------- SET CORRECT!!!
                status = Status.OBJECT_DETECTION
                if config.auto:
                    transition = True 
                else:
                    transition = False #<---------------- False, wait for trigger
            else:
                continue
                

        elif status == Status.OBJECT_DETECTION:
            # ======================================
            #   Detect object9
            # ======================================
            if transition:
                print("")
                print(f">> Step 1: Checking for Bottle <<")
                print(f"=================================")
                transition = False
            
            if (object_msg is None) or (len(object_msg.detections)<=0):
                print(f"\r[INFO] [ ] Cup Detected ", end='', flush=True)
                continue
            else:
                #print(f"len {len(object_msg.detections)}")
                for i in reversed(range(len(object_msg.detections))):
                    #print(f"Object Detected: {classes[object_msg.detections[i].results[0].id]}\n")
                    #print(f"Object at: {object_msg.detections[i].bbox.center}\n")
                    # Crope element otuside interesting area or with bad id
                    bound = [config.area_x_min,config.area_x_max,       # boundaries[x0,x1,y0,y1] of interesting area
                             config.area_y_min,config.area_y_max]
                    who = config.desired_classes                        # desired classes
                    if utils.pop_prediction(object_msg.detections[i],classes,
                                            bound,who):
                        #print(f"cropped: {classes[object_msg.detections[i].results[0].id]}\n")
                        object_msg.detections.pop(i)
                
                if len(object_msg.detections)>0:
                    if utils.check_detections(object_msg.detections, classes, ['cup']):
                        print(f"\r[INFO] [✔️] Cup Detected    ", end='', flush=True)
                        if config.auto:
                            status = Status.APPROACH
                            transition = True
                    elif utils.check_detections(object_msg.detections, classes, ['bottle']):
                        print(f"\r[INFO] [✖️] Bottle Detected ", end='', flush=True)
                    else:
                        print(f"\r[INFO] [  ] Cup Detected ", end='', flush=True)
                else:
                    print(f"\r[INFO] [  ] Cup Detected ", end='', flush=True)
                    #utils.print_detections(object_msg.detections,classes)


                object_msg = None
        
        elif status == Status.APPROACH:
            # ======================================
            #   Approach
            # ======================================
            if transition:
                print("")
                print(f">> Step 2: Approaching the object <<")
                print(f"====================================")
                print(f"[INFO] [ ] Reached", end='', flush=True)
                for target_name in config.approaching_trajectory_names:
                    target = robot_controller.joint_control_points[target_name]
                    robot_controller.go_to(target)
                
                print(f"\r[INFO] [✔️] Reached")
                status = Status.GRASP

                if not(config.auto):
                    transition = False #<---------------- False, wait for trigger
            else:
                continue
        elif status == Status.GRASP:
            # ======================================
            #   Grasp object
            # ======================================
            if transition:
                print("")
                print(f">> Step 3: Grasping the object <<")
                print(f"=================================")
                print(f"[INFO] [ ] Grasped", end='', flush=True)
                robot_controller.grasp(config.sensing_closure)
                rospy.sleep(5)    
                print(f"\r[INFO] [✔️] Grasped")

                
                status = Status.SENSE 
                if not(config.auto):
                    transition = False #<---------------- False, wait for trigger
                    
            else:
                continue

        elif status == Status.SENSE:
            # ======================================
            #   Sensing object
            # ======================================
            if transition:
                print("")
                print(f">> Step 4: Temperature Sensing <<")
                print(f"=================================")
                temperature_msgs = []
                utils.sleep_with_progressbar(config.sensing_duration, rospy.sleep,"Sensing")

                # Classify with Sensor
                # Choose relative action <-------------------------------------
                if len(temperature_msgs)==0: 
                    rospy.logwarn('no temperature recieved')
                    continue
               
                output = temperature_msgs[-1].temperature
                #if output>0.76:
                #    output_class = "coldest" # coldest, warm, hottest 
                #    #output_class = "warm" # coldest, warm, hottest
                #elif output>0.65 and output<0.76:
                #    output_class = "warm" # coldest, warm, hottest 
                #elif output<.65:
                #    output_class = "hottest" # coldest, warm, hottest 
                    #output_class = "warm" # coldest, warm, hottest
                output_class = detect_class(output)

                #print(f"[INFO] Found {output_class} object: {output:.3f}/{-(output-1.72)/0.02:.1f} [V/°C]")
                print(f"[INFO] Found {output_class} object")

                temperature_msgs = None
                
                status = Status.MOVE
                if not(config.auto):
                    transition = False #<---------------- False, wait for trigger
            else:
                continue
        
        
        elif status == Status.MOVE:
            # ======================================
            #   Move object
            # ======================================
            if transition:
                print("")
                print(f">> Step 5: Moving the object <<")
                print(f"===============================")
                print(f"[INFO] [ ] Reinforce Grasp", end='', flush=True)
                robot_controller.grasp(config.grasping_closure)
                rospy.sleep(5)    
                print(f"\r[INFO] [✔️] Reinforce Grasp")
                print(f"[INFO] [ ] Positioning object", end='', flush=True)
                
                # set destination
                release_dest = set_destination(output_class, config)
                #if output_class == 'coldest':
                #    release_dest = config.release_cold_trajectory_names
                #elif output_class == 'warm':
                #    release_dest = config.release_warm_trajectory_names
                #elif output_class == 'hottest':
                #    release_dest = config.release_hot_trajectory_names
                #else:
                #    release_dest = config.release_warm_trajectory_names

                # execute traject
                for target_name in release_dest:
                    target = robot_controller.joint_control_points[target_name]
                    robot_controller.go_to(target)
                print(f"\r[INFO] [✔️] Positioning object")
                print(f"[INFO] [ ] Release object", end='', flush=True)
                robot_controller.release()
                rospy.sleep(5)    
                print(f"\r[INFO] [✔️] Release object")

                status = Status.IDLE
                if not(config.auto):
                    transition = False #<---------------- False, wait for trigger
            else:
                continue # <---------------- Comment to wait for trigger


if __name__ == "__main__":
    main()
