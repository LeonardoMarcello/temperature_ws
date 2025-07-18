#!/usr/bin/env python3

import rospy
import numpy as np
import copy
import os
from enum import Enum, auto
from datetime import datetime
import csv

from sensor_msgs.msg import Temperature


from temperature_ros import RobotController as RC
from temperature_ros import MaterialClassifier as MC


# =============================================
# Config classes to set experiment parameters
# =============================================  
class Config():
    supervised = True
    # Notes to add in csv
    #material = 'alluminio, thick 1mm, temp 38C'
    #material = 'plastica, thick 2mm, temp 38C'
    #material = 'vetro, thick 2mm, temp 38C'
    #material = 'legno, thick 3mm, temp 38C'
    #material = 'legno, thick 5mm, temp 38C'
    material = 'video'

    # hard coded trajectoy
    rest_trajectory_names = ['up']
    approaching_trajectory_names = ['up']
    sensing_trajectory_names = ['close','down']

    # closure percentage 
    sensing_closure = 0.0

    # time of sensing [s] 
    sensing_duration = 40

    control_points_path = f'src/temperature_ros/config/exp_materials.yaml'
    csv_saves_name_path = "workdir/exp_materials_" + datetime.now().strftime("%d%m%Y%H%M%S") + ".csv"
    dataset_path = f'src/temperature_ros/config/net/data/train/'
    weights_path = f'src/temperature_ros/config/net/weights/'
    weight_name = f'material_classifier_model_v3_alpv'

# =============================================================================
# Callback
# =============================================================================  
temperature_msgs = []
delta_temperature_msgs = []
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


def dt_callback(msg:Temperature):
    global delta_temperature_msgs
    if delta_temperature_msgs is not None: delta_temperature_msgs.append(msg)


csv_file = None
def cleanup():
    csv_file.close()

# =============================================================================
# Main
# =============================================================================  
def main():
    global temperature_msgs, delta_temperature_msgs, csv_file

    rospy.init_node("temperature_experiment", anonymous=True)
    rospy.on_shutdown(cleanup)

    rate = rospy.Rate(10)  # Loop rate [Hz]
    exp = 1
    samples = 0
    config = Config()

    save_flag = False

    # open csv
    csv_file = open(config.csv_saves_name_path, mode='w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Experiment', 'Timestamp', 'temperature [C]', 'delta_t [C/s]', 'notes'])  # Header
    rospy.loginfo(f"Created file: {config.csv_saves_name_path}")
    
    # Robot COntroller
    robot_controller = RC.RobotController(control_points = config.control_points_path)

    # Classification manager
    dataset = MC.MaterialDataset(config.dataset_path, which=0)
    net = MC.MaterialClassifier(dataset,config.weights_path, num_features = 2)
    rospy.loginfo(F"Net Device: {net.device}")

    #lstm = LSTMNet(num_features=1, num_hidden1=125, num_hidden2=100, num_classes=len(dataset.class_names))
    #lstm = torch.load(config.weights_path + config.weight_name + '.pt', weights_only=False)
    #lstm.to(self.device)
    #lstm = lstm.float() 
    #net.lstm = lstm
    net.load(config.weight_name)
    rospy.loginfo(F"Loaded net weights: {config.weights_path + config.weight_name}")
    x1 = np.empty((0,1))
    x2 = np.empty((0,1))
    x_t = np.empty((0,1))

    #callback triggers 
    temperature_sub = rospy.Subscriber("/temperature", Temperature, t_callback)
    delta_temperature_sub = rospy.Subscriber("/delta_temperature", Temperature, dt_callback)

    while not rospy.is_shutdown():


        # sleep rate
        rate.sleep()

        # 1. Wait for user input to start log
        if not save_flag:
            if config.supervised:
                choice = input(f"============ Experiment {exp}. Press `Enter` to sense or `q` to exit\n")
                if choice == 'q':
                    csv_file.close()
                    return
            else:
                if exp > 15:
                    # Check num of experiment
                    csv_file.close()
                    return
                try:
                    # Read sensed T
                    T = temperature_msgs[-1].temperature
                except Exception as e:
                    rospy.logwarn(e)
                    T = -100
                if T <= 1.05:
                    # Check if returned to Tamb
                    continue
            
            # Approach 
            for target_name in config.approaching_trajectory_names:
                target = robot_controller.joint_control_points[target_name]
                robot_controller.go_to(target)
            # Sense 
            for target_name in config.sensing_trajectory_names:
                target = robot_controller.joint_control_points[target_name]
                robot_controller.go_to(target)
            # start buffering measurements
            temperature_msgs = []
            delta_temperature_msgs = []
            samples = 0
            t0 = rospy.Time.now().to_sec()
            print(f'start recording for {config.sensing_duration} seconds')
            # start recording
            save_flag = True


        # 2. Save
        if save_flag: 
            try:
                # Load last measurements
                T = temperature_msgs[-1].temperature
                dT = delta_temperature_msgs[-1].temperature

                # Write to file (append mode)
                row = [exp, rospy.Time.now(), T, dT, config.material]
                csv_writer.writerow(row)
                samples +=1
                
                # Append measurent for prediction
                x1 = np.vstack((x1, T)) 
                x2 = np.vstack((x2, dT)) 
                x_t = np.vstack((x_t, 1e-9*rospy.Time.now().to_nsec()))

            except Exception as e:
                rospy.logwarn(e)
            
            # Stop savings
            t = rospy.Time.now().to_sec()
            if t-t0 > config.sensing_duration: 
                save_flag = False
                print(f'Exp {exp}, recorded {samples} samples')
                exp += 1
                
                # Classify                
                try:
                    (pred, score) = net.predict(np.hstack([x1,x2,x_t]))
                    print(f'Prediction {pred}: {net.dataset_manager.class_names[pred]}')
                    print(score)
                except Exception as e:
                    rospy.logwarn(e)

                # Flush out measurements
                x1 = np.empty((0,1))
                x2 = np.empty((0,1))
                x_t = np.empty((0,1))

                # Return up 
                for target_name in config.approaching_trajectory_names:
                    target = robot_controller.joint_control_points[target_name]
                    robot_controller.go_to(target)
        

if __name__ == "__main__":
    main()
