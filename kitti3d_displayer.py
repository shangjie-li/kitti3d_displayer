import os
import numpy as np
import open3d
import matplotlib.pyplot as plt
from pathlib import Path

import box_utils
import calibration_kitti
import object3d_kitti

classes = ['Car', 'Pedestrian', 'Cyclist']

COLORS = (
    (0, 255, 0), # Car
    (255, 0, 0), # Pedestrian
    (0, 255, 255) # Cyclist
)

def get_o3d_box(box3d, color=[1.0, 0.0, 0.0]):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        box3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    box3d = np.array(box3d)
    assert len(box3d.shape) == 1
    corners3d = box_utils.boxes_to_corners_3d(np.array([box3d]))[0]
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
        [0, 5], [1, 4], # heading
    ])
    colors = [color for i in range(len(edges))]
    
    lines = open3d.geometry.LineSet()
    lines.points = open3d.utility.Vector3dVector(corners3d)
    lines.lines = open3d.Vector2iVector(edges)
    lines.colors = open3d.utility.Vector3dVector(colors)
    return lines

def get_points_in_fov(pts, calib, img_h, img_w):
    """
    Args:
        pts (ndarray, [N, 4]): x, y, z, intensity
        calib (Calibration)
        img_h (int)
        img_w (int)
        
    Returns:
        pts_fov (ndarray, [N', 4]): x, y, z, intensity
    """
    pts_rect = calib.lidar_to_rect(pts[:, 0:3])
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect) # [N, 2], [N]
    
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_w)
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_h)
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    
    pts_fov = pts[pts_valid_flag]
    return pts_fov

def get_colored_points_in_fov(pts, calib, img):
    """
    Args:
        pts (ndarray, [N, 4]): x, y, z, intensity
        calib (Calibration)
        img (ndarray, [img_h, img_w, 3]): BGR image
        
    Returns:
        pts_fov_colored (ndarray, [N', 7]): x, y, z, intensity, r, g, b
    """
    
    img_h, img_w, _ = img.shape
    pts_fov = get_points_in_fov(pts, calib, img_h, img_w) # [N', 4]
    pts_rect = calib.lidar_to_rect(pts_fov[:, 0:3])
    pts_fov_img, pts_fov_rect_depth = calib.rect_to_img(pts_rect) # [N', 2], [N']
    
    pts_fov_img = pts_fov_img.astype(np.int)
    pts_rgb = img[pts_fov_img[:, 1], pts_fov_img[:, 0], :] # [N', 3]
    pts_fov_colored = np.concatenate([pts_fov, pts_rgb], axis=1)
    return pts_fov_colored

if __name__ == '__main__':
    root_path = '/home/lishangjie/data/KITTI3D/kitti3d/training'
    image_id = '000211'
    
    img_path = os.path.join(root_path, 'image_2', image_id + '.png')
    assert Path(img_path).exists(), 'Not Found: %s' % img_path
    img = plt.imread(img_path)
    
    pts_path = os.path.join(root_path, 'velodyne', image_id + '.bin')
    assert Path(pts_path).exists(), 'Not Found: %s' % pts_path
    pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 4)
    
    calib_path = os.path.join(root_path, 'calib', image_id + '.txt')
    assert Path(calib_path).exists(), 'Not Found: %s' % calib_path
    calib = calibration_kitti.Calibration(calib_path)
    
    label_path = os.path.join(root_path, 'label_2', image_id + '.txt')
    assert Path(label_path).exists(), 'Not Found: %s' % label_path
    obj_list = object3d_kitti.get_objects_from_label(label_path)
    
    annotations = {}
    annotations['name'] = np.array([obj.cls_type for obj in obj_list])
    annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
    annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
    annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
    annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
    annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
    annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
    annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
    annotations['score'] = np.array([obj.score for obj in obj_list])
    annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)
    
    num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
    num_gt = len(annotations['name'])
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    
    loc = annotations['location'][:num_objects]
    dims = annotations['dimensions'][:num_objects]
    rots = annotations['rotation_y'][:num_objects]
    loc_lidar = calib.rect_to_lidar(loc)
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2
    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='points', width=800, height=600)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.asarray([0.4, 0.4, 0.4])
    
    pts_fov_colored = get_colored_points_in_fov(pts, calib, img) # [N', 7]
    points = pts_fov_colored[:, :3] # [N', 3]
    colors = pts_fov_colored[:, 4:] # [N', 3]
    colors = np.array(([1.0, 1.0, 1.0]))[None, :].repeat(points.shape[0], axis=0)
    
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points)
    pts.colors = open3d.utility.Vector3dVector(colors)
    vis.add_geometry(pts)
    
    gt_boxes = gt_boxes_lidar
    for j in range(gt_boxes.shape[0]):
        try:
            name = annotations['name'][j]
            class_idx = classes.index(name)
            color = [c / 255.0 for c in COLORS[class_idx]]
            box = get_o3d_box(gt_boxes[j, :7], color=color)
            vis.add_geometry(box)
        except:
            pass
    
    vis.run()
    vis.destroy_window()
