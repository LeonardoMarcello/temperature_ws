import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import os

import copy
from tf.transformations import quaternion_matrix
#from moveit_msgs.msg import CollisionObject
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from std_msgs.msg import Int32
import glob
import sys

def add_collision_object(publisher, center, dimensions, object_id, shape_type="BOX"):
    collision_object = CollisionObject()
    collision_object.header.frame_id = "world"
    collision_object.id = object_id
    primitive = SolidPrimitive()
    if shape_type == "BOX":
        primitive.type = SolidPrimitive.BOX
    elif shape_type == "PLANE":
        primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [dimensions[0], dimensions[1], dimensions[2]]
    pose = Pose()
    pose.position.x = center[0]
    pose.position.y = center[1]
    pose.position.z = center[2]
    collision_object.primitives.append(primitive)
    collision_object.primitive_poses.append(pose)
    collision_object.operation = CollisionObject.ADD
    publisher.publish(collision_object)

def load_and_publish_point_clouds(count_publisher):
    #rospy.init_node('point_cloud_collision_publisher', anonymous=True)
    publisher = rospy.Publisher("/collision_object", CollisionObject, queue_size=10)
    rospy.sleep(1)
    all_points = []
    # all_points_down = []
    count_publisher.publish(len(all_points))

    rospy.sleep(5)
    #rospy.signal_shutdown("Creazione degli oggetti di collisione completata")

def add_environment_collision_objects(publisher, file_name):
    base_center = [0.0, 0.0, 0.0]
    base_dimensions = [2.0, 1.0, 0.02]
    add_collision_object(publisher, base_center, base_dimensions, f"{file_name}_base", shape_type="PLANE")
    
    base_center_2 = [-0.3, 0.0, 0.5]
    base_dimensions_2 = [0.05, 1.0, 1.5]
    add_collision_object(publisher, base_center_2, base_dimensions_2, f"{file_name}_muro_dietro", shape_type="PLANE")
    
    base_center_3 = [0.0, 0.2, 0.3]
    base_dimensions_3 = [0.1, 0.1, 0.5]
    add_collision_object(publisher, base_center_3, base_dimensions_3, f"{file_name}_camera_1", shape_type="PLANE")
    
    base_center_4 = [1.2, -0.33, 0.3]
    base_dimensions_4 = [0.1, 0.10, 0.5]
    add_collision_object(publisher, base_center_4, base_dimensions_4, f"{file_name}_camera_2", shape_type="PLANE")

    base_center_5 = [0.0, -0.5, 0.3]
    base_dimensions_5 = [1, 0.01, 1]
    add_collision_object(publisher, base_center_5, base_dimensions_5, f"{file_name}_muro_sinistro", shape_type="PLANE")

    base_center_6 = [-0.10, 0.25, 0.3]
    base_dimensions_6 = [0.15, 0.15, 0.5]
    add_collision_object(publisher, base_center_6, base_dimensions_6, f"{file_name}_cubo_destro", shape_type="PLANE")

    base_center_7 = [0.0, 0.43, 0.3]
    base_dimensions_7 = [1, 0.01, 1]
    add_collision_object(publisher, base_center_7, base_dimensions_7, f"{file_name}_muro_destro", shape_type="PLANE")

    # base_center_8 = [0.45, 0, 0.015]
    # base_dimensions_8 = [0.52, 0.28, 0.04]
    # add_collision_object(publisher, base_center_8, base_dimensions_8, f"{file_name}_rialzo", shape_type="PLANE")

def plan_grasp_trajectory():
    
    # move_group.go(pose_finale, wait=True)
    #READY 
    user_input = input("Premi 's' per tornare la condizione di riposo: ")
    if user_input.lower() == 's':
        print("Hai premuto 's'. Il programma continua...")
    else:
        print("Input non valido. Il programma si interrompe.")                

    # We get the joint values from the group and change some of the values:
    joint_goal = move_group.get_current_joint_values()
    joint_goal[0] = 0.0
    joint_goal[1] = -0.785
    joint_goal[2] = 0.0
    joint_goal[3] = -2.356
    joint_goal[4] = 0.0
    joint_goal[5] = 1.571
    joint_goal[6] = 0.785

    move_group.go(joint_goal, wait=True)

    #APPROACH PLACE 
    user_input = input("Premi 's' per arrivare all'approccio place: ")
    if user_input.lower() == 's':
        print("Hai premuto 's'. Il programma continua...")
    else:
        print("Input non valido. Il programma si interrompe.")                

    # We get the joint values from the group and change some of the values:
    joint_goal = move_group.get_current_joint_values()
    joint_goal[0] = -0.084
    joint_goal[1] = -0.309
    joint_goal[2] = -0.183
    joint_goal[3] = -2.357
    joint_goal[4] = -0.102
    joint_goal[5] = 2.106
    joint_goal[6] = 0.561

    move_group.go(joint_goal, wait=True)

    #PLACE 
    user_input = input("Premi 's' per arrivare al place: ")
    if user_input.lower() == 's':
        print("Hai premuto 's'. Il programma continua...")
    else:
        print("Input non valido. Il programma si interrompe.")                
        return
    
    # We get the joint values from the group and change some of the values:
    joint_goal = move_group.get_current_joint_values()
    joint_goal[0] = -0.100
    joint_goal[1] = 0.216
    joint_goal[2] = -0.190
    joint_goal[3] = -2.518
    joint_goal[4] = -0.027
    joint_goal[5] = 2.787
    joint_goal[6] = 0.461

    move_group.go(joint_goal, wait=True)

    #READY 
    user_input = input("Premi 's' per tornare la condizione di riposo: ")
    if user_input.lower() == 's':
        print("Hai premuto 's'. Il programma continua...")
    else:
        print("Input non valido. Il programma si interrompe.")
        return              

    # We get the joint values from the group and change some of the values:
    joint_goal = move_group.get_current_joint_values()
    joint_goal[0] = 0.0
    joint_goal[1] = -0.785
    joint_goal[2] = 0.0
    joint_goal[3] = -2.356
    joint_goal[4] = 0.0
    joint_goal[5] = 1.571
    joint_goal[6] = 0.785

    move_group.go(joint_goal, wait=True)

    user_input = input("Premi 's' per fare il prossimo grasp : ")
    if user_input.lower() == 's':
        print("Hai premuto 's'. Il programma continua...")
    else:
        print("Input non valido. Il programma si interrompe.")
        return

def remove_collision_object(publisher, object_id):
    collision_object = CollisionObject()
    collision_object.header.frame_id = "world"
    collision_object.id = object_id
    collision_object.operation = CollisionObject.REMOVE
    publisher.publish(collision_object)

def remove_collision_objects_from_folder(publisher, pcd_folder, pcd_folder_down):
    # Ottieni la lista dei file pcd nella cartella principale e in quella down
    #rospy.init_node('point_cloud_collision_publisher', anonymous=True)
    publisher = rospy.Publisher("/collision_object", CollisionObject, queue_size=10)
    rospy.sleep(1)

    pcd_files = sorted(glob.glob(os.path.join(pcd_folder, '*.pcd')))
    pcd_files_down = sorted(glob.glob(os.path.join(pcd_folder_down, '*.pcd')))

    # Rimuovi gli oggetti di collisione associati ai file nella cartella principale
    for pcd_file in pcd_files:
        print("Percorso del file:", pcd_file)
        cloud_points, file_name = load_point_cloud(pcd_file, percentuale)
        for idx in range(len(cloud_points)):
            object_id = f"{file_name}_{idx}"
            remove_collision_object(publisher, object_id)
            rospy.loginfo("Oggetto {} rimosso dalla scena".format(object_id))
            print(f"Verifica: Oggetto {object_id} rimosso dalla scena")

    # Rimuovi gli oggetti di collisione associati ai file nella cartella down
    for pcd_file_down in pcd_files_down:
        cloud_points_down, file_name_down = load_point_cloud(pcd_file_down, percentuale)
        for idx in range(len(cloud_points_down)):
            object_id = f"{file_name_down}_{idx}"
            remove_collision_object(publisher, object_id)
            rospy.loginfo("Oggetto {} rimosso dalla scena".format(object_id))
            print(f"Verifica: Oggetto {object_id} rimosso dalla scena")

if __name__ == "__main__":
    rospy.init_node("main_node", anonymous=True)

    count_publisher = rospy.Publisher("/collision_object_count", Int32, queue_size=10)
    publisher = rospy.Publisher("/collision_object", CollisionObject, queue_size=10)
    load_and_publish_point_clouds(count_publisher)

    file_name = "ambiente"  # Puoi cambiare il nome del file come preferisci    
    add_environment_collision_objects(publisher, file_name)

    user_input = input("Premi 's' per continuare con la traiettoria di grasp: ")
    if user_input.lower() == 's':
        print("Hai premuto 's'. Il programma continua...")
    else:
        print("Input non valido. Il programma si interrompe.")

    moveit_commander.roscpp_initialize(sys.argv)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "panda_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    display_trajectory_publisher = rospy.Publisher(
        "/move_group/display_planned_path",
        moveit_msgs.msg.DisplayTrajectory,
        queue_size=20,
    )

    plan_grasp_trajectory()
    user_input = input("Premi 's' per continuare con la cancellazione: ")
    if user_input.lower() == 's':
        print("Hai premuto 's'. Il programma continua...")
    else:
        print("Input non valido. Il programma si interrompe.")
    # Rimuovi gli oggetti di collisione dalle cartelle specificate



# bisogna caricare le collisioni down più le collisioni del cluster di interesse e non quelle degli altri.
#    --> oppure si potrebbe migliiorare la condizione di approach dove la collisione degli altri cluster trovati avviene quando l'approach è stata svolta.

