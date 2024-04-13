import open3d as o3d
import numpy as np
import os

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'source/')

# Load the point cloud from the PLY file
point_cloud = o3d.io.read_point_cloud(src_dir + "femur_shell.ply")
point_cloud.paint_uniform_color([0.5, 0.5, 0.5])
pcd_center = point_cloud.get_center()

# Original STL
stl = o3d.io.read_triangle_mesh(src_dir + "femur_shell.stl")
stl.compute_vertex_normals()
stl.scale(0.001, center=[0, 0, 0])
stl_center = stl.get_center()



# PLAN 1
# Points (x,y,z) in m
p1 = np.array([13.98485, -36.4029, 20.07205]) / 1000.00
p2 = np.array([12.98173, -37.48651, 17.947]) / 1000.00
p3 = np.array([16.12607, -38.20582, 18.66463]) / 1000.0

# PLAN 2
# Points (x,y,z) in m
p1 = np.array([2.899343, 4.66568, 22.83373]) / 1000.00
p2 = np.array([1.10593, 7.09621, 22.13024]) / 1000.00
p3 = np.array([0.203026, 3.98506, 21.95038]) / 1000.0




# Create three spheres at the specified points
sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.0005)
sphere1.compute_vertex_normals()
sphere1.paint_uniform_color([1, 0, 0])
sphere1.translate(p1)

sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.0005)
sphere2.compute_vertex_normals()
sphere2.paint_uniform_color([0, 1, 0])
sphere2.translate(p2)

sphere3 = o3d.geometry.TriangleMesh.create_sphere(radius=0.0005)
sphere3.compute_vertex_normals()
sphere3.paint_uniform_color([0, 0, 1])
sphere3.translate(p3)

# Create a Triangle Mesh with the three points
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector([p1, p2, p3])
mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.9, 0.9, 0])
normal =  np.asarray(mesh.vertex_normals)[0]
# print(normal)


# Create Coordinate Frame
pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.005, origin=[0, 0, 0])

# Transform the Coordinate Frame
actual_normal = -normal
z_axis = np.array([0, 0, 1])
rotation_axis = np.cross(z_axis, actual_normal)
rotation_axis /= np.linalg.norm(rotation_axis)
angle = np.arccos(np.dot(z_axis, actual_normal) / (np.linalg.norm(z_axis) * np.linalg.norm(actual_normal)))

R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = p3

pose.transform(T)

print(T)

# Additonal Rotation (if needed)
theta = -np.pi / 2
axis = R[:, 2]
R_ = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * theta)
pose.rotate(R_, center=p3)

# Create Global Coordinate Frame
global_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.001, origin=[0, 0, 0])
sc_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.001, origin=stl_center)




# Visualize wrt point cloud
scene = [point_cloud, sphere1, sphere2, sphere3, mesh, pose]
o3d.visualization.draw_geometries(scene)

# Visualize wrt STL
# scene = [sphere1, sphere2, sphere3, mesh, pose, global_pose, sc_pose, stl]
# o3d.visualization.draw_geometries(scene)

