import open3d as o3d
import os
import sys
import dataset_location
from registration import Estimator, Refiner
import numpy as np
import time

class PointCloudVisualizerWithRegistration:
    def __init__(self, directories, source_file, markers=[]):
        # Files
        self.dataset_directory = directories[0]
        self.target_directory = directories[1]
        self.dataset_files = [file for file in sorted(os.listdir(self.dataset_directory)) if file.endswith('.ply')]
        self.target_files = [file for file in sorted(os.listdir(self.target_directory)) if file.endswith('.ply')]
        self.source_path = source_file


        if not self.target_files:
            print("No .ply files found in the target directory.")
            sys.exit(0)
        elif not self.dataset_files:
            print("No .ply files found in the dataset directory.")
            sys.exit(0)

        # Registration
        self.estimator = Estimator('centroid')
        self.refiner = Refiner('ransac_icp')

        

        self.current_index = 0
        self.markers = markers
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        
    def load_point_cloud(self, directory,file):
        if directory == self.dataset_directory:
            file_path = os.path.join(directory, file)
        else:
            file_path = os.path.join(directory, file)
        return o3d.io.read_point_cloud(file_path)
    
    def update_visualization(self):
        self.vis.clear_geometries()
        print("Current Index: ", self.current_index)


        current_file = self.target_files[self.current_index]
        self.current_file_num = current_file.split("bone")[1].split(".")[0]

        print("Current File: ", current_file)

        for file in self.dataset_files:
            if self.current_file_num in file:
                dataset_file = file
                break
        else:
            print("No matching file found in the dataset directory.")
            sys.exit(0)



        self.original_cloud = self.load_point_cloud(self.dataset_directory,dataset_file)
        self.target_cloud = self.load_point_cloud(self.target_directory,current_file)
        self.source_cloud = o3d.io.read_point_cloud(self.source_path)
        self.target_cloud.paint_uniform_color([0, 0.651, 0.929])
        self.source_cloud.paint_uniform_color([1, 0, 0])

        # Point Cloud Center
        if 'pcd_center' in self.markers:
            center = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            center.compute_vertex_normals()
            center.paint_uniform_color([0, 0, 1])
            center.translate(self.target_cloud.get_center())
            self.vis.add_geometry(center)

        self.register()



        # self.vis.add_geometry(self.original_cloud)
        self.vis.add_geometry(self.target_cloud)
        self.vis.add_geometry(self.source_cloud)
        self.vis.update_renderer()
        # self.vis.capture_screen_image("results/reg_result_{0}.png".format(time.time_ns()))

    
    def next_cloud(self):
        self.vis.capture_screen_image("results/reg_result_{0}.png".format(self.current_file_num))
        self.current_index = (self.current_index + 1) % len(self.target_files)
        self.update_visualization()
    
    def previous_cloud(self):
        self.current_index = (self.current_index - 1) % len(self.target_files)
        self.update_visualization()

    def refresh_cloud(self):
        self.update_visualization()

    
    def run(self):
        self.vis.create_window()
        self.vis.register_key_callback(262, lambda vis: self.next_cloud())  # Right arrow key
        self.vis.register_key_callback(263, lambda vis: self.previous_cloud())  # Left arrow key
        self.vis.register_key_callback(32, lambda vis: self.refresh_cloud())  # Space key


        view_control = self.vis.get_view_control()
        view_control.set_front([-0.00022986335794028144, 0.18784944702287554, 0.9821978071732989])
        view_control.set_lookat([-0.023245651523880972, 0.004731096545583302, -1.6438119959420225])
        view_control.set_up([-0.0026792206420369336, 0.9821941922094379, -0.1878493826628165])
        view_control.set_zoom(0.25999999999999956)

        view_control.camera_local_translate(forward=0, right=0.5, up=0)

        self.update_visualization()
        self.vis.run()

    def register(self):
        print("Registering...")

        init_transformation = self.estimator.estimate(self.source_cloud, self.target_cloud)

        self.source_cloud = self.source_cloud.voxel_down_sample(voxel_size=0.005)
        self.target_cloud = self.target_cloud.voxel_down_sample(voxel_size=0.005)

        # noise = np.random.normal(0, 0.02, (4,4))
        # init_transformation += noise
        # self.source_cloud.transform(init_transformation)


        transform = self.refiner.refine(self.source_cloud, self.target_cloud, init_transformation)
        print("Transformation Matrix: ")
        print(transform)

        self.source_cloud.transform(transform)


if __name__ == "__main__":
    full_directory_path = dataset_location.DATASET_PATH
    target_directory_path = "filtered"
    source_file_path = "source/femur.ply"
    directories = [full_directory_path, target_directory_path]
    markers = ['obb','pcd_center']
    visualizer = PointCloudVisualizerWithRegistration(directories,source_file_path,markers)
    visualizer.run()
