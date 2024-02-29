import numpy as np
import open3d as o3d
from copy import deepcopy
import matplotlib.pyplot as plt

class PointCloudProcessingPipeline:
    def __init__(self, pcd_path, x_thresh, y_thresh, z_thresh):
        self.pcd_path = pcd_path
        self.x_thresh = x_thresh
        self.y_thresh = y_thresh
        self.z_thresh = z_thresh
        self.pcd = None
        self.center = None
        self.debug = False

    def load_point_cloud(self):
        self.pcd = o3d.io.read_point_cloud(self.pcd_path)
        self.center = self.pcd.get_center()

        # o3d.visualization.draw_geometries([self.pcd], window_name="Original Point Cloud")

    def crop_point_cloud(self):
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=self.center - np.array([self.x_thresh, self.y_thresh, self.z_thresh]),
            max_bound=self.center + np.array([self.x_thresh, self.y_thresh, self.z_thresh])
        )
        self.pcd = self.pcd.crop(bounding_box)

        if self.debug:
            o3d.visualization.draw_geometries([self.pcd], window_name="Cropped Point Cloud")

    def segment_plane(self, iter=1):
        for _ in range(iter):
            plane_model, inliers = self.pcd.segment_plane(distance_threshold=0.005,
                                                        ransac_n=3,
                                                        num_iterations=1000)
            [plane_a, plane_b, plane_c, plane_d] = plane_model
            print(f"Plane equation: {plane_a}x + {plane_b}y + {plane_c}z + {plane_d} = 0")

            inlier_cloud = self.pcd.select_by_index(inliers)
            outlier_cloud = self.pcd.select_by_index(inliers, invert=True)
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            self.pcd = outlier_cloud

            if self.debug:
                o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name="Outlier Cloud")



    def cluster_dbscan(self, eps=0.02, min_points=10,color=False):
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(self.pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
        
        max_label = labels.max()
        print(f"Point cloud has {max_label + 1} clusters")

        if color:
            self.visualize_clusters(labels)
        else:
            cluster_index = np.argsort(np.bincount(labels[labels >= 0]))[::-1][0]
            self.filter_cluster(labels, cluster_index)

    def visualize_clusters(self, labels):
        max_label = labels.max()
        pcd_clusters = deepcopy(self.pcd)
        colors = plt.get_cmap("tab20")(np.linspace(0, 1, max_label + 1))
        new_colors = np.zeros_like(np.asarray(pcd_clusters.colors))
        
        for i in range(max_label + 1):
            new_colors[labels == i] = colors[i][:3]
        new_colors[labels == -1] = [0.5, 0.5, 0.5]
        
        pcd_clusters.colors = o3d.utility.Vector3dVector(new_colors)
        self.pcd.colors = o3d.utility.Vector3dVector(new_colors)
        
        if self.debug:
            o3d.visualization.draw_geometries([pcd_clusters], window_name="Point Cloud Clusters")

    def filter_cluster(self, labels, cluster_index):
        filtered_points = np.asarray(self.pcd.points)[labels == cluster_index]
        filtered_colors = np.asarray(self.pcd.colors)[labels == cluster_index]

        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

        self.pcd = filtered_pcd
        
        if self.debug:
            o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Point Cloud")


    def outlier_removal(self):
        cl, ind = self.pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        total_points = np.asarray(self.pcd.points).shape[0]
        print(f"Removed {total_points - np.asarray(cl.points).shape[0]} outliers from the point cloud")
        
        self.pcd = self.pcd.select_by_index(ind)
        


    def run(self,debug=False):
        self.debug = debug
        self.load_point_cloud()
        # self.crop_point_cloud()
        self.segment_plane(iter=1)
        # Adjust eps and min_points as needed
        eps = 0.003
        min_points = 10
        self.cluster_dbscan(eps, min_points)
        self.outlier_removal()
        self.cluster_dbscan(0.005, 20, color=True)

        return self.pcd

def main():
    pcd_path = "/home/warra/bone_dataset_1/ply/bone03.ply"
    x_thresh = 0.15
    y_thresh = 0.3
    z_thresh = 0.3

    pipeline = PointCloudProcessingPipeline(pcd_path, x_thresh, y_thresh, z_thresh)
    pipeline.run()

if __name__ == "__main__":
    main()
