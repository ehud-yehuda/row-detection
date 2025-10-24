import numpy as np
import cv2

class CameraCalibration:
    def __init__(
            self,
            fx, fy, cx, cy,
            image_width, image_height,
            T_cam_lidar=None
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.image_width = image_width
        self.image_height = image_height
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        if T_cam_lidar is None:
            self.T_cam_lidar = np.eye(4, dtype=np.float32)
        else:
            self.T_cam_lidar = T_cam_lidar


class LiDARIntegrator:
    def __init__(self, calibration):
        self.calib = calibration

    def transform_points_to_camera_frame(self, points_lidar):
        if points_lidar.shape[1] == 4:
            xyz = points_lidar[:, :3]
        else:
            xyz = points_lidar
        N = xyz.shape[0]
        points_hom = np.hstack([xyz, np.ones((N, 1))])

        points_cam_hom = (self.calib.T_cam_lidar @ points_hom.T).T

        points_camera = points_cam_hom[:, :3]

        return points_camera

    def project_points_to_image(self, points_camera):
        N = points_camera.shape[0]

        valid_depth = points_camera[:, 2] > 0

        points_normalized = points_camera[:, :2] / points_camera[:, 2:3]
        pixels_hom = (self.calib.K @ np.vstack([
            points_normalized.T,
            np.ones(N)
        ])).T

        pixels = pixels_hom[:, :2]
        depths = points_camera[:, 2]

        valid_x = (pixels[:, 0] >= 0) & (pixels[:, 0] < self.calib.image_width)
        valid_y = (pixels[:, 1] >= 0) & (pixels[:, 1] < self.calib.image_height)
        valid_mask = valid_depth & valid_x & valid_y

        return pixels, depths, valid_mask

    def create_depth_map(
            self,
            points_lidar,
            max_depth=50.0
    ):
        points_camera = self.transform_points_to_camera_frame(points_lidar)

        pixels, depths, valid = self.project_points_to_image(points_camera)

        h, w = self.calib.image_height, self.calib.image_width
        depth_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.int32)

        valid_pixels = pixels[valid].astype(np.int32)
        valid_depths = depths[valid]

        depth_filter = valid_depths < max_depth
        valid_pixels = valid_pixels[depth_filter]
        valid_depths = valid_depths[depth_filter]

        for (u, v), d in zip(valid_pixels, valid_depths):
            depth_map[v, u] += d
            count_map[v, u] += 1

        valid_mask = count_map > 0
        depth_map[valid_mask] = depth_map[valid_mask] / count_map[valid_mask]

        return depth_map, valid_mask

    def get_depth_at_bbox(self, bbox, depth_map, method='median'):
        if len(bbox) == 4:
            x, y, w, h = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
        else:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        h, w = depth_map.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        region = depth_map[y1:y2, x1:x2]

        valid_depths = region[region > 0]

        if len(valid_depths) == 0:
            return None

        # Compute depth based on method
        if method == 'mean':
            return float(np.mean(valid_depths))
        elif method == 'median':
            return float(np.median(valid_depths))
        elif method == 'min':
            return float(np.min(valid_depths))
        elif method == 'center':
            cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
            if depth_map[cy, cx] > 0:
                return float(depth_map[cy, cx])
            else:
                return float(np.median(valid_depths))
        else:
            return float(np.median(valid_depths))

    def bbox_to_3d_position(self, bbox, depth_map, bbox_format='xywh'):
        depth = self.get_depth_at_bbox(bbox, depth_map, method='median')

        if depth is None or depth <= 0:
            return None

        if bbox_format == 'xywh':
            x, y, w, h = bbox
            cx = x + w / 2.0
            cy = y + h / 2.0
        else:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

        X = (cx - self.calib.cx) * depth / self.calib.fx
        Y = (cy - self.calib.cy) * depth / self.calib.fy
        Z = depth

        return np.array([X, Y, Z], dtype=np.float32)

    def visualize_depth_map(self, depth_map, max_depth=None):
        if max_depth is None:
            max_depth = np.max(depth_map[depth_map > 0])

        depth_normalized = np.clip(depth_map / max_depth * 255, 0, 255).astype(np.uint8)

        vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        vis[depth_map == 0] = [0, 0, 0]

        return vis



if __name__ == "__main__":
    print("\nExample 1: Camera Calibration Setup")

    calib = CameraCalibration(
        fx=800.0,  # Focal length X
        fy=800.0,  # Focal length Y
        cx=640.0,  # Principal point X (image center)
        cy=360.0,  # Principal point Y
        image_width=1280,
        image_height=720
    )

    print("Camera intrinsic matrix K:")
    print(calib.K)

    print("Example 2: Generate Mock LiDAR Point Cloud")

    # Create random points in front of camera
    num_points = 1000
    points_lidar = np.random.randn(num_points, 3).astype(np.float32)
    points_lidar[:, 0] = points_lidar[:, 0] * 5 + 5
    points_lidar[:, 1] = points_lidar[:, 1] * 2
    points_lidar[:, 2] = points_lidar[:, 2] * 1 + 1.5

    print("Generated {} LiDAR points".format(num_points))
    print("Point cloud range:")
    print("  X: [{:.2f}, {:.2f}]".format(points_lidar[:, 0].min(), points_lidar[:, 0].max()))
    print("  Y: [{:.2f}, {:.2f}]".format(points_lidar[:, 1].min(), points_lidar[:, 1].max()))
    print("  Z: [{:.2f}, {:.2f}]".format(points_lidar[:, 2].min(), points_lidar[:, 2].max()))

    print("Example 3: Create Depth Map")

    integrator = LiDARIntegrator(calib)

    depth_map, valid_mask = integrator.create_depth_map(
        points_lidar,
        max_depth=20.0
    )

    print("Depth map shape: {}".format(depth_map.shape))
    print("Valid pixels: {} / {} ({:.1f}%)".format(
        np.sum(valid_mask),
        valid_mask.size,
        100.0 * np.sum(valid_mask) / valid_mask.size
    ))
    print("Depth range: [{:.2f}, {:.2f}] meters".format(
        depth_map[valid_mask].min(),
        depth_map[valid_mask].max()
    ))

    print("Example 4: Get Depth for Bounding Box")

    bbox = np.array([500, 300, 80, 120])

    depth = integrator.get_depth_at_bbox(bbox, depth_map, method='median')

    if depth is not None:
        print("Bounding box: x={}, y={}, w={}, h={}".format(*bbox))
        print("Median depth: {:.2f} meters".format(depth))
    else:
        print("No valid depth found for bbox")

    print("Example 5: Bbox to 3D Position")

    position_3d = integrator.bbox_to_3d_position(bbox, depth_map, bbox_format='xywh')

    if position_3d is not None:
        print("3D Position (camera frame):")
        print("  X: {:.2f} m (lateral)".format(position_3d[0]))
        print("  Y: {:.2f} m (vertical)".format(position_3d[1]))
        print("  Z: {:.2f} m (depth)".format(position_3d[2]))
    else:
        print("Failed to compute 3D position")

    print("Example 6: Visualize Depth Map")

    depth_vis = integrator.visualize_depth_map(depth_map, max_depth=15.0)
    print("Depth visualization shape: {}".format(depth_vis.shape))