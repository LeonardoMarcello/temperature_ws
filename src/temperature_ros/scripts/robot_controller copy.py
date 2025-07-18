#!/usr/bin/env python3

import rospy

import moveit_commander
import sys
from scipy.spatial.transform import Rotation as R

import actionlib

import numpy as np
import tf

from franka_msgs.msg import ErrorRecoveryAction, ErrorRecoveryActionGoal

from dynamic_reconfigure.server import Server

import std_msgs.msg
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Temperature

from std_srvs.srv import Empty, EmptyResponse


REFERENCE_FRAME = 'base_link'     # nome frame target base robot
EE_FRAME = 'hand_link'     # nome frame target base robot


try:
    from math import pi, tau, dist, fabs, cos
except:  
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi
    
    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


class RobotController(object):
    def __init__(self):
        # super(RobotController, self).__init__()

        moveit_commander.roscpp_initialize(sys.argv)
        robot = moveit_commander.RobotCommander()
        robot_group_name = "panda_arm"
        move_group_robot = robot.get_group(robot_group_name)
        robot_planning_frame = move_group_robot.get_planning_frame()
        move_group_robot.set_end_effector_link(EE_FRAME)
        ee_link = move_group_robot.get_end_effector_link() 

        scene = moveit_commander.PlanningSceneInterface()
        self.robot = robot
        self.scene = scene
        self.move_group_robot = move_group_robot
        self.robot_planning_frame = robot_planning_frame
        self.ee_link = ee_link

        self.target
        self.measured_temperature
        # Subscribers and Services
        rospy.Service('/grasp', Empty, self.grasp_callback)
        rospy.Subscriber('/set_target', PoseStamped, self.set_target_callback)
        rospy.Subscriber('/temperature', Temperature, self.temperature_callback)

        self.robot_group_name = robot_group_name

        self.error_recovery = actionlib.SimpleActionClient('/franka_control/error_recovery', ErrorRecoveryAction)
        self.error_recovery.wait_for_server()
        

    
    def dyn_rec_callback(self, config, level):
        rospy.loginfo("""Reconfigure Request: {table_height} """.format(**config))
        self.table_height = config.table_height
        return config

    def prepare_robot(self):
        self.move_group_robot.set_named_target('ready')
        self.move_group_robot.go(wait=True)
        self.move_group_robot.stop()
        self.move_group_robot.clear_pose_targets()

    def execute_plan(self, plan):
        move_group = self.move_group_robot
        executed = move_group.execute(plan, wait=True)
        if not executed:
            print("Trajectory not executed ", executed)
            goal = ErrorRecoveryActionGoal()
            goal.header = std_msgs.msg.Header()

            self.error_recovery.send_goal(goal)
            wait = self.error_recovery.wait_for_result(rospy.Duration(5.0))
            if not wait:
                rospy.logerr("ErrorRecoveryActionGoal Action server not available!")
                

    def go_to_pose(self, target_pose):
        move_group = self.move_group_robot
        move_group.set_pose_target(target_pose)        
        
        plan = False
        attempt = 1
        MAX_ATTEMPTS = 3
        while not plan and attempt <= MAX_ATTEMPTS:
            plan = move_group.plan()              
            plan = move_group.go()
            stop = False
            if plan:
                move_group.stop()
                move_group.clear_pose_targets()
            else:
                print("Pose not reached ")
                stop = True
                goal = ErrorRecoveryActionGoal()
                goal.header = std_msgs.msg.Header()

                self.error_recovery.send_goal(goal)
                wait = self.error_recovery.wait_for_result(rospy.Duration(5.0))
                if not wait:
                    rospy.logerr("ErrorRecoveryActionGoal Action server not available!")
            print("Attempt N: ", attempt)
            attempt = attempt + 1  	      
            
        return plan, stop
    

    def set_target(self, position, orientation):
        target = PoseStamped()
        target.pose.position.x = position[0]
        target.pose.position.y = position[0]
        target.pose.position.z = position[0]
        
        target.pose.orientation.x = orientation[0]
        target.pose.orientation.y = orientation[1]
        target.pose.orientation.z = orientation[2]
        target.pose.orientation.w = orientation[3]
        
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time(0)
        header.frame_id = REFERENCE_FRAME
        target.header = header
        return target
    
    def set_target_callback(self, msg):
        position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        orientation = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        self.target = self.set_target(position, orientation)
        print("New Target at: " + str(position) + " " + str(orientation))
        return
    
    def grasp_callback(self, req):
        print("Going to Target")
        self.go_to_pose(self.target)
        return EmptyResponse()
    
    def temperature_callback(self, msg):
        self.measured_temperature = msg.temperature
        return

# =============================================================================
# Main
# =============================================================================  
def main():
    rospy.init_node("robot_controller", anonymous=True)
    robot_controller_node = RobotController()

    goal = ErrorRecoveryActionGoal()
    goal.header = std_msgs.msg.Header()

    robot_controller_node.error_recovery.send_goal(goal)
    wait = robot_controller_node.error_recovery.wait_for_result(rospy.Duration(5.0))

    while not rospy.is_shutdown():
        robot_controller_node.experiment += 1
        choice = input(f"============ Press:\n"
                       "             `h` to return to home pose\n"
                       "             `q` to quite...")
        
       
        # ======================================
        #   Return to ready
        # ======================================
        if choice == "h" or choice == "H":
            print("returning to ready")
            robot_controller_node.move_group_robot.set_named_target('ready')
            executed = robot_controller_node.move_group_robot.go(wait=True)
            robot_controller_node.move_group_robot.stop()
            robot_controller_node.move_group_robot.clear_pose_targets()

        # ======================================
        #   Quit
        # ======================================
        elif choice == "q" or choice == "Q":            
            exit(1)
        else:
            print("wrong input. Repeat please.")


if __name__ == "__main__":
    main()
