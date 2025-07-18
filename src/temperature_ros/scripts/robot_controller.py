#!/usr/bin/env python3

import rospy
import numpy as np

from temperature_ros import RobotController as RC


# =============================================================================
# Main
# =============================================================================  
def main():
    rospy.init_node("robot_controller", anonymous=True)
    #robot_controller = RC.RobotController(control_points = f'src/temperature_ros/config/test.yaml')
    robot_controller = RC.RobotController(control_points = f'src/temperature_ros/config/exp_materials.yaml')
    handstatus = 0
    grasp_closure = 0.53
    firm_grasp_closure = 0.63

    while not rospy.is_shutdown():
        choice = input(f"============ Press:\n"
                       f"             `g` to grasp/release ({grasp_closure}). press 'G' to frim grasp ({firm_grasp_closure})\n"
                       "             `m` to move in target pose\n"
                       "             `h` to return to home pose\n"
                       "             `s` to store current pose as control point\n"
                       "             `l` to get current control points list\n"
                       "             `r` to read control point from yaml\n"
                       "             `w` to write control point in yaml\n"
                       "             `q` to quite\n"
                       ">> ")
        
       
        # ======================================
        #   Return to ready
        # ======================================
        if choice == "h" or choice == "H":
            print("returning to ready")
            robot_controller.back_to_ready()
        
        # ======================================
        #   Grasp/Release
        # ======================================
        elif choice == "g":
            if handstatus==0: robot_controller.grasp(grasp_closure)    # closure percentage
            elif handstatus==1: robot_controller.release()
            handstatus=np.mod(handstatus+1,2)
            print(f'handstatus {handstatus}')
        elif choice == "G":
            robot_controller.grasp(firm_grasp_closure)    # closure percentage
            
        # ======================================
        #   Store current position to ready
        # ======================================
        elif choice == "s" or choice == "S":
            print("Save current Position")
            name = input(f"Insert desired name:\n")
            config = robot_controller.add_current_control_point(name)
            print(f"Added new position: '{name}'\n{config}")
        
        # ======================================
        #   Print control points
        # ======================================
        elif choice == "l" or choice == "L":
            print("Control points dictionary:")
            robot_controller.print_control_point()
        # ======================================
        #   Read/Write control points
        # ======================================
        elif choice == "r" or choice == "R":
            name = input(f"Insert name of the yaml file:\n")
            robot_controller.load_control_point_as_yaml(f'src/temperature_ros/config/{name}.yaml', override = False)
            print("Control points loaded")
            
        elif choice == "w" or choice == "W":
            name = input(f"Insert name of the yaml file:\n")
            robot_controller.save_control_point_as_yaml(f'src/temperature_ros/config/{name}.yaml')
            print("Control points stored")


        # ======================================
        #   Move to target
        # ======================================
        elif choice == "m" or choice == "M":
            print("Insert target name or 'q' to abort:")
            cp = robot_controller.joint_control_points
            for target_name in cp.keys():
                print(f'>> {target_name}')
            target_name = input(f"\n>> ")
            
            if target_name == 'q': continue
            
            try:
                target = cp[target_name]
                robot_controller.go_to(target)
            except KeyError:
                print(f"{target_name} has not been stored")

        # ======================================
        #   Quit
        # ======================================
        elif choice == "q" or choice == "Q":            
            exit(1)
        else:
            print("wrong input. Repeat please.")


if __name__ == "__main__":
    main()
