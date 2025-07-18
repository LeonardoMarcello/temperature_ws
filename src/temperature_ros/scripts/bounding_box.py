#! /usr/bin/env python3

import open3d as o3d
import numpy as np
import pandas as pd
import os

# Calcola le bounding box (OBB e AABB)
def compute_bounding_boxes(pcd):
    # Calcola la Axis-Aligned Bounding Box (AABB)
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)  # Rosso per AABB

    # Calcola la Oriented Bounding Box (OBB)
    obb = pcd.get_oriented_bounding_box()
    obb.color = (0, 1, 0)  # Verde per OBB

    return aabb, obb

# Visualizza il point cloud con le bounding box
def visualize(pcd, aabb, obb):
    o3d.visualization.draw_geometries([pcd, aabb, obb],
                                      zoom=0.8,
                                      front=[-0.5, -0.5, -0.5],
                                      lookat=[0, 0, 0],
                                      up=[0, 1, 0])

# Carica pointcloud, taglia tra threshold[0] e threshold[1] e applica rotazione
def load_point_and_rotate_cloud(pcd_file, rotation_matrix, threshold_x, threshold_y, threshold_z, save=False):
    # Carica il file PCD
    cloud = o3d.io.read_point_cloud(pcd_file)
    # Converti il punto cloud in un DataFrame pandas
    
    df = pd.DataFrame(np.asarray(cloud.points), columns=['x', 'y', 'z'])
    # Applica i threshold lungo le dimensioni x, y e z
    df = df[(df['x'] >= threshold_x[0]) & (df['x'] <= threshold_x[1])]
    df = df[(df['y'] >= threshold_y[0]) & (df['y'] <= threshold_y[1])]
    df = df[(df['z'] >= threshold_z[0]) & (df['z'] <= threshold_z[1])]

    # Applica una rotazione
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(df.values)
    points.rotate(rotation_matrix)
    o3d.visualization.draw_geometries([points])

    # Salva la nuvola di punti come file PCD
    if (save):
        output_folder = "src/temperature_ros/pcd/mug3"
        filename = os.path.join(output_folder, f"pcd_cut.pcd")
        o3d.io.write_point_cloud(filename, points)
        # Salva il point cloud come file PCD
        print("File PCD salvato in " + filename)
    return points




# main
if __name__ == "__main__":
    # Genera la point cloud
    pcd = "src/temperature_ros/pcd/mug3.pcd"
    threshold_x = [0.1, 0.80]
    threshold_y = [-0.3, 0.3]
    threshold_z = [0.04, 0.35]
    rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    ###### Caricamento della point cluod #######
    cloud = load_point_and_rotate_cloud(pcd, rotation_matrix, threshold_x, threshold_y, threshold_z)    

    # Calcola le bounding box
    aabb, obb = compute_bounding_boxes(cloud)
    
    center = aabb.get_center()
    (L, W, H) = aabb.max_bound - aabb.min_bound
    #R = obb.R
    #(L, W, H) = obb.extent
    print(center)
    #print(R)
    print(L,W,H)
    print("proporzioni: " + str(H/W))

    # Visualizza il risultato
    visualize(cloud, aabb, obb)