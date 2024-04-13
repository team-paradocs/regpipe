import os
import open3d as o3d

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'source')

# Get a list of all STL files in the current directory
stl_files = [file for file in os.listdir(src_dir) if file.lower().endswith('.stl')]
stl_files = [os.path.join(src_dir, file) for file in stl_files]

# Iterate over each STL file
for stl_file in stl_files:
    # Load the STL file
    mesh = o3d.io.read_triangle_mesh(stl_file)

    # Scale the mesh by 0.001
    mesh.scale(0.001, center=[0, 0, 0])
    print(f"Loaded {stl_file}")
    print(f"Center: {mesh.get_center()}")


    # Convert the mesh to a point cloud using Poisson sampling
    point_cloud = mesh.sample_points_poisson_disk(5000)

    # Save the point cloud as a PLY file
    ply_file = os.path.splitext(stl_file)[0] + '.ply'
    o3d.io.write_point_cloud(ply_file, point_cloud)

    o3d.visualization.draw_geometries([point_cloud])

    print(f"Saved {ply_file}")


print("Conversion complete!")