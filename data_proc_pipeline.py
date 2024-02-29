import open3d as o3d
import os
import sys
import json

from proc_pipeline import PointCloudProcessingPipeline
import dataset_location


class PointCloudVisualizer:
    def __init__(self, directory):
        self.directory = directory
        self.ply_files = [file for file in sorted(os.listdir(directory)) if file.endswith('.ply')]
        if not self.ply_files:
            print("No .ply files found in the directory.")
            sys.exit(0)
        else:
            print(self.ply_files)
        self.current_index = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        
    def load_point_cloud(self, index):
        file_path = os.path.join(self.directory, self.ply_files[index])
        return o3d.io.read_point_cloud(file_path)
    
    def update_visualization(self):
        self.vis.clear_geometries()
        # point_cloud = self.load_point_cloud(self.current_index)

        file_path = os.path.join(self.directory, self.ply_files[self.current_index])
        x_thresh = 0.15
        y_thresh = 0.3
        z_thresh = 0.3

        pipeline = PointCloudProcessingPipeline(file_path, x_thresh, y_thresh, z_thresh)
        self.point_cloud = pipeline.run()
        self.vis.add_geometry(self.point_cloud)
        self.vis.update_renderer()

    
    def next_cloud(self):
        self.current_index = (self.current_index + 1) % len(self.ply_files)
        self.update_visualization()
    
    def previous_cloud(self):
        self.current_index = (self.current_index - 1) % len(self.ply_files)
        self.update_visualization()

    def save_cloud(self):
        file_name = self.ply_files[self.current_index].split(".")[0]
        o3d.io.write_point_cloud("filtered/filtered_{}.ply".format(file_name), self.point_cloud)
        print("Saved filtered point cloud to filtered/filtered_{}.ply".format(file_name))
    
    def run(self):
        self.vis.create_window()
        self.vis.register_key_callback(262, lambda vis: self.next_cloud())  # Right arrow key
        self.vis.register_key_callback(263, lambda vis: self.previous_cloud())  # Left arrow key
        self.vis.register_key_callback(264, lambda vis: self.save_cloud()) # Down arrow key
        self.update_visualization()
        self.vis.run()

if __name__ == "__main__":
    directory_path = dataset_location.DATASET_PATH
    visualizer = PointCloudVisualizer(directory_path)
    visualizer.run()
