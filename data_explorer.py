import open3d as o3d
import os
import sys
import json
import dataset_location

class PointCloudVisualizer:
    def __init__(self, directory,markers=[]):
        self.directory = directory
        self.ply_files = [file for file in os.listdir(directory) if file.endswith('.ply')]
        if not self.ply_files:
            print("No .ply files found in the directory.")
            sys.exit(0)
        self.current_index = 0
        self.markers = markers
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        
    def load_point_cloud(self, index):
        file_path = os.path.join(self.directory, self.ply_files[index])
        return o3d.io.read_point_cloud(file_path)
    
    def update_visualization(self):
        self.vis.clear_geometries()
        self.point_cloud = self.load_point_cloud(self.current_index)

        # Shift the point cloud to the origin
        self.point_cloud = self.point_cloud.translate(-self.point_cloud.get_center())

        # Custom Markers
        # Oriented Bounding Box
        if 'obb' in self.markers:
            obb = self.point_cloud.get_oriented_bounding_box()
            obb.color = (1, 0, 0)
            self.vis.add_geometry(obb)

        # Point Cloud Center
        if 'pcd_center' in self.markers:
            center = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            center.compute_vertex_normals()
            center.paint_uniform_color([0, 0, 1])
            center.translate(self.point_cloud.get_center())
            self.vis.add_geometry(center)


        self.vis.add_geometry(self.point_cloud)
        self.vis.update_renderer()

    
    def next_cloud(self):
        self.current_index = (self.current_index + 1) % len(self.ply_files)
        self.update_visualization()
    
    def previous_cloud(self):
        self.current_index = (self.current_index - 1) % len(self.ply_files)
        self.update_visualization()

    
    def run(self):
        self.vis.create_window()
        self.vis.register_key_callback(262, lambda vis: self.next_cloud())  # Right arrow key
        self.vis.register_key_callback(263, lambda vis: self.previous_cloud())  # Left arrow key

        self.update_visualization()
        self.vis.run()

if __name__ == "__main__":
    directory_path = dataset_location.DATASET_PATH
    directory_path = "filtered"
    markers = ['obb','pcd_center']
    visualizer = PointCloudVisualizer(directory_path,markers)
    visualizer.run()
