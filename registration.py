import numpy as np
import open3d as o3d


class Estimator:
    def __init__(self, method='centroid'):
        self.method = method


    def estimate(self, source, target):
        if self.method == 'centroid':
            return self.centroid(source, target)
        else:
            print("Invalid estimation method.")
            return None
        

    def centroid(self, source, target):
        '''
        Centroid-based transformation estimation
        '''
        source_center = source.get_center()
        target_center = target.get_center()

        # Translate source to origin
        translation_to_origin = np.eye(4)
        translation_to_origin[0:3, 3] = -source_center

        # Create a rotation matrix
        roll, pitch, yaw = np.radians([0, 0, 90])  # Convert degrees to radians
        rotation = o3d.geometry.get_rotation_matrix_from_xyz((roll, pitch, yaw))
        rotation_4x4 = np.eye(4)  # Expand to 4x4 matrix
        rotation_4x4[0:3, 0:3] = rotation  # Set the top-left 3x3 to the rotation matrix

        # Translate back to target's position
        translation_back = np.eye(4)
        translation_back[0:3, 3] = target_center

        # Combine transformations
        transformation = translation_back @ rotation_4x4 @ translation_to_origin

        return transformation

class Refiner:
    def __init__(self, method='p2pl_icp'):
        self.method = method


    
    def refine(self, source, target, transformation=np.eye(4)):
        if self.method == 'p2pl_icp':
            return self.p2pl_icp(source, target, transformation)
        elif self.method == 'robust_p2pl_icp':
            return self.robust_p2pl_icp(source, target, transformation)
        else:
            print("Invalid refinement method.")
            return None
        


    def p2pl_icp(self, source, target, transformation):
        '''
        Open3D Point-to-Plane ICP
        '''
        threshold = 0.02
        max_iter = 2000


        # Compute normals for source and target point clouds
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))


        reg_result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )

        print(reg_result)

        return reg_result.transformation
    
    def robust_p2pl_icp(self, source, target, transformation):
        '''
        Open3D Point-to-Plane ICP with Robust Kernels
        '''

        threshold = 0.02
        max_iter = 2000

        # Compute normals for source and target point clouds
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
        p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)


        reg_result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, transformation,
            p2l,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )

        print(reg_result)

        return reg_result.transformation




    

    