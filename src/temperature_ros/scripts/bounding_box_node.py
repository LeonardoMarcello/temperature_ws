#! /usr/bin/env python3

import open3d as o3d
import numpy as np
import pandas as pd
import os

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

import rospy
from std_srvs.srv import Empty, EmptyResponse

# ========================================================================================
class Target:
    def __init__(self, name, cloud = None, threshold_x = None, threshold_y= None, threshold_z=None ):
        self.name = name
        if threshold_x is None or threshold_y is None or threshold_z is None:
            self.limits = None
        else:
            self.limits = [threshold_x, threshold_y, threshold_z]
        self.points = o3d.geometry.PointCloud()
        self.aabb = None
        self.obb = None
        if cloud is not None:
            self.update_cloud(cloud)
            self.compute_bounding_boxes()
    

    def update_cloud(self, cloud):
        df = pd.DataFrame(np.asarray(cloud.points), columns=['x', 'y', 'z'])
        if self.limits is not None:
            # Applica i threshold lungo le dimensioni x, y e z
            df = df[(df['x'] >= self.limits[0][0]) & (df['x'] <= self.limits[0][1])]
            df = df[(df['y'] >= self.limits[1][0]) & (df['y'] <= self.limits[1][1])]
            df = df[(df['z'] >= self.limits[2][0]) & (df['z'] <= self.limits[2][1])]
        # updating point cloud
        self.points.points = o3d.utility.Vector3dVector(df.values)


    def compute_bounding_boxes(self):
        # Calcola la Axis-Aligned Bounding Box (AABB)
        self.aabb = self.points.get_axis_aligned_bounding_box()
        # Calcola la Oriented Bounding Box (OBB)
        self.obb = self.points.get_oriented_bounding_box()

    def get_center(self):
            return self.aabb.get_center()
    
    def get_size(self):
        (L, W, H) = self.aabb.max_bound - self.aabb.min_bound
        return (L,W,H)

    def get_ratio(self):
        (L,W,H) = self.get_size()
        return H/W
    
    def visualize(self):
        self.aabb.color = (1, 0, 0)  # Rosso per AABB
        self.obb.color = (0, 1, 0)  # Verde per OBB
        o3d.visualization.draw_geometries([self.points, self.aabb, self.obb],
                                        zoom=0.8,
                                        front=[-0.5, -0.5, -0.5],
                                        lookat=[0, 0, 0],
                                        up=[0, 1, 0])

# ========================================================================================
# Initialization
# ========================================================================================
targets = []
o3d_cloud = o3d.geometry.PointCloud()
stopped = False

output_folder = os.path.join(os.getcwd(), "src/temperature_ros/pcd/")
pcd_save_name = "experiment.pcd"

pcd_topic = "/cloud_pcd"

def init_framework():
    # Target 1: Bottiglia calda
    target1 = Target("Bottiglia calda", threshold_x = [0.1, 0.80], threshold_y = [0.05, 0.3], threshold_z = [0.04, 0.35])
    targets.append(target1)
    # Target 2: sfera
    target2 = Target("Sfera", threshold_x = [0.1, 0.80], threshold_y = [-0.3, 0.1], threshold_z = [0.04, 0.35])
    targets.append(target2)
    # Target 3: bottiglia fredda
    target3 = Target("Bottiglia calda", threshold_x = [0.1, 0.80], threshold_y = [-0.3, 0.3], threshold_z = [0.04, 0.35])
    targets.append(target3)
    return

# ========================================================================================
def pc_callback(msg):
    if stopped:
        return
    stopped = True
    point_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    # Convert to NumPy array
    points_np = np.array(point_list, dtype=np.float32)
    # Create Open3D point cloud
    o3d_cloud.points = o3d.utility.Vector3dVector(points_np)
    # udate objects cloud
    for i in range(len(targets)):
        targets[i].update_cloud(o3d_cloud)
    stopped = False


def compute_targets_callback(req):   
    stopped = True
    print("Computing targets...")   
    for i in range(len(targets)):
        try:
            targets[i].compute_bounding_boxes()  
            targets[i].visualize()
            print("Name: " + targets[i].name)
            print("Center [mm]: " + str(targets[i].get_center()))
            print("Ratio H/W: " + str(targets[i].get_ratio()))
        except Exception as e:
            print(e)
    stopped = False

    return EmptyResponse()
def save_pcd_callback(req):
    stopped = True
    print("Saving point cloud")
    
    # Applica una rotazione
    #o3d_cloud.rotate(rotation_matrix)

    filename = os.path.join(output_folder, pcd_save_name)
    o3d.io.write_point_cloud(filename, o3d_cloud)
    stopped = False
    
    return EmptyResponse()

def bounding_box_node():
    rospy.init_node('bounding_box_node')
    rospy.Subscriber(pcd_topic, PointCloud2, pc_callback)
    rospy.Service('/compute_targets', Empty, compute_targets_callback)
    rospy.Service('/save_pcd', Empty, save_pcd_callback)

    rospy.spin()

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    init_framework()            # select targets and its rough location
    bounding_box_node()         # run bounding box node

