import numpy as np
import cv2


class PathSkeletonExtractor:
    def __init__(
            self,
            min_path_area=1000,
            skeleton_method='morphological'
    ):
        self.min_path_area = min_path_area
        self.skeleton_method = skeleton_method

    def extract_skeleton(self, path_mask):
        if np.sum(path_mask) < self.min_path_area:
            return np.zeros_like(path_mask), np.array([])

        path_mask = (path_mask > 0).astype(np.uint8)

        if self.skeleton_method == 'morphological':
            skeleton = self._morphological_skeleton(path_mask)
        elif self.skeleton_method == 'distance_transform':
            skeleton = self._distance_transform_skeleton(path_mask)
        else:
            raise ValueError("Unknown skeleton method: {}".format(self.skeleton_method))

        if skeleton is None:
            skeleton_points = None
        else:
            skeleton_points = np.argwhere(skeleton > 0)[:, ::-1]

        return skeleton, skeleton_points

    def _morphological_skeleton(self, mask, num_of_iteration=1000):
        skeleton = np.zeros_like(mask)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        converge = False
        iteration_counter = 0

        while iteration_counter < num_of_iteration:
            iteration_counter += 1
            # Open
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)
            # Subtract
            temp = cv2.subtract(mask, opened)
            # Erode
            eroded = cv2.erode(mask, element)
            skeleton = cv2.bitwise_or(skeleton, temp)
            mask = eroded.copy()

            if cv2.countNonZero(mask) == 0:
                converge = True
                break

        if not converge:
            skeleton = None

        return skeleton

    def _distance_transform_skeleton(self, mask):
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        threshold = np.percentile(dist[mask > 0], 70)
        skeleton = (dist > threshold).astype(np.uint8)

        skeleton = cv2.ximgproc.thinning(skeleton) if hasattr(cv2, 'ximgproc') else skeleton

        return skeleton

    def sort_skeleton_points(self, skeleton_points, start_from='bottom'):
        if len(skeleton_points) == 0:
            return skeleton_points

        if start_from == 'bottom':
            idx = np.argsort(skeleton_points[:, 1])[::-1]
        elif start_from == 'top':
            idx = np.argsort(skeleton_points[:, 1])
        elif start_from == 'left':
            idx = np.argsort(skeleton_points[:, 0])
        elif start_from == 'right':
            idx = np.argsort(skeleton_points[:, 0])[::-1]
        else:
            raise ValueError("Unknown start_from: {}".format(start_from))

        return skeleton_points[idx]


class CurveFitter:
    def __init__(
            self,
            method='polynomial',
            poly_degree=2
    ):
        self.method = method
        self.poly_degree = poly_degree

    def fit_curve(self, points):
        if len(points) < 3:
            return None, None

        if self.method == 'polynomial':
            return self._fit_polynomial(points)
        elif self.method == 'line':
            return self._fit_line(points)
        elif self.method == 'spline':
            return self._fit_spline(points)
        else:
            raise ValueError("Unknown method: {}".format(self.method))

    def _fit_polynomial(self, points):
        x = points[:, 0]
        y = points[:, 1]

        coeffs = np.polyfit(y, x, self.poly_degree)  # Fit x as function of y
        poly = np.poly1d(coeffs)

        params = {
            'type': 'polynomial',
            'coefficients': coeffs,
            'degree': self.poly_degree,
            'direction': 'x_from_y'  # x = f(y)
        }

        def curve_func(y_val):
            return poly(y_val)

        return params, curve_func

    def _fit_line(self, points):
        x = points[:, 0]
        y = points[:, 1]

        A = np.vstack([y, np.ones(len(y))]).T
        b_coef, a_coef = np.linalg.lstsq(A, x, rcond=None)[0]

        params = {
            'type': 'line',
            'slope': b_coef,
            'intercept': a_coef,
            'direction': 'x_from_y'
        }

        def curve_func(y_val):
            return a_coef + b_coef * y_val

        return params, curve_func

    def _fit_spline(self, points):
        try:
            from scipy import interpolate
        except ImportError:
            print("Warning: scipy not available, falling back to polynomial")
            return self._fit_polynomial(points)

        x = points[:, 0]
        y = points[:, 1]

        # Sort by y
        idx = np.argsort(y)
        x_sorted = x[idx]
        y_sorted = y[idx]

        # Fit spline
        tck = interpolate.splrep(y_sorted, x_sorted, s=0)

        params = {
            'type': 'spline',
            'tck': tck,
            'direction': 'x_from_y'
        }

        def curve_func(y_val):
            return interpolate.splev(y_val, tck)

        return params, curve_func


class MaskBasedDirectionCalculator:
    def __init__(
            self,
            skeleton_method='distance_transform',
            curve_method='polynomial',
            poly_degree=2
    ):
        self.skeleton_extractor = PathSkeletonExtractor(
            skeleton_method=skeleton_method
        )
        self.curve_fitter = CurveFitter(
            method=curve_method,
            poly_degree=poly_degree
        )

    def calculate_direction_from_mask(
            self,
            path_mask,
            camera_calibration,
            depth_map=None,
            tractor_position=None,
            look_ahead_pixels=200
    ):
        if tractor_position is None:
            tractor_position = np.zeros(3)

        skeleton, skeleton_points = self.skeleton_extractor.extract_skeleton(path_mask)

        if len(skeleton_points) < 3 or skeleton is None:
            return None, None, 0.0

        skeleton_points = self.skeleton_extractor.sort_skeleton_points(
            skeleton_points,
            start_from='bottom'
        )

        curve_params, curve_func = self.curve_fitter.fit_curve(skeleton_points)

        if curve_func is None:
            return None, None, 0.0

        h, w = path_mask.shape
        current_y = h - 1
        current_x = curve_func(current_y)

        target_y = max(0, current_y - look_ahead_pixels)
        target_x = curve_func(target_y)

        direction_2d = np.array([
            target_x - current_x,
            target_y - current_y
        ])

        norm_2d = np.linalg.norm(direction_2d)
        if norm_2d < 1e-6:
            return None, None, 0.0

        direction_2d = direction_2d / norm_2d

        if depth_map is not None:
            direction_vector_3d, target_point_3d = self._convert_to_3d(
                current_x, current_y,
                target_x, target_y,
                depth_map,
                camera_calibration,
                tractor_position
            )
        else:
            direction_vector_3d = np.array([direction_2d[0], 0.0, -direction_2d[1]])
            direction_vector_3d = direction_vector_3d / np.linalg.norm(direction_vector_3d)
            target_point_3d = tractor_position + direction_vector_3d * 5.0

        confidence = self._calculate_confidence(
            skeleton_points,
            curve_params,
            path_mask
        )

        return direction_vector_3d, target_point_3d, confidence

    def _convert_to_3d(
            self,
            current_x, current_y,
            target_x, target_y,
            depth_map,
            calib
    ):
        h, w = depth_map.shape
        cy, cx = int(current_y), int(np.clip(current_x, 0, w - 1))
        ty, tx = int(target_y), int(np.clip(target_x, 0, w - 1))

        current_depth = depth_map[cy, cx] if depth_map[cy, cx] > 0 else 5.0
        target_depth = depth_map[ty, tx] if depth_map[ty, tx] > 0 else 5.0

        current_3d = self._backproject_point(cx, cy, current_depth, calib)
        target_3d = self._backproject_point(tx, ty, target_depth, calib)

        direction = target_3d - current_3d
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None, None

        direction = direction / norm

        return direction, target_3d

    def _backproject_point(self, u, v, depth, calib):
        X = (u - calib.cx) * depth / calib.fx
        Y = (v - calib.cy) * depth / calib.fy
        Z = depth
        return np.array([X, Y, Z])

    def _calculate_confidence(self, skeleton_points, curve_params, path_mask):
        confidence = 0.0

        if len(skeleton_points) > 100:
            confidence += 0.3
        elif len(skeleton_points) > 50:
            confidence += 0.2
        elif len(skeleton_points) > 20:
            confidence += 0.1

        path_area = np.sum(path_mask)
        total_area = path_mask.size
        coverage = path_area / float(total_area)
        if coverage > 0.2:
            confidence += 0.3
        elif coverage > 0.1:
            confidence += 0.2
        elif coverage > 0.05:
            confidence += 0.1

        if curve_params is not None:
            confidence += 0.2

        if len(skeleton_points) > 10:
            y_range = skeleton_points[:, 1].max() - skeleton_points[:, 1].min()
            h = path_mask.shape[0]
            if y_range > 0.5 * h:
                confidence += 0.2

        return min(confidence, 1.0)

    def visualize(self, image, path_mask, skeleton, direction_vector=None):
        vis = image.copy()

        mask_overlay = np.zeros_like(vis)
        mask_overlay[path_mask > 0] = [0, 255, 0]
        vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

        vis[skeleton > 0] = [255, 0, 0]

        if direction_vector is not None:
            h, w = image.shape[:2]
            start = (w // 2, h - 50)
            # Project direction to 2D (use x and z components)
            end = (
                int(start[0] + direction_vector[0] * 100),
                int(start[1] - direction_vector[2] * 100)
            )
            cv2.arrowedLine(vis, start, end, (0, 0, 255), 3, tipLength=0.3)

        return vis



if __name__ == "__main__":
    print("\nExample 1: Generate Mock Path Mask")

    h, w = 720, 1280
    path_mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        center_x = w // 2 + int(30 * np.sin(y / 100.0))
        path_width = 200

        x1 = max(0, center_x - path_width // 2)
        x2 = min(w, center_x + path_width // 2)
        path_mask[y, x1:x2] = 1

    print("Path mask shape: {}".format(path_mask.shape))
    print("Path pixels: {:,}".format(np.sum(path_mask)))

    print("Example 2: Extract Path Skeleton")

    extractor = PathSkeletonExtractor(skeleton_method='distance_transform')
    skeleton, skeleton_points = extractor.extract_skeleton(path_mask)

    print("Skeleton points: {}".format(len(skeleton_points)))
    print("Skeleton shape: {}".format(skeleton.shape))

    print("Example 3: Fit Curve to Skeleton")

    sorted_points = extractor.sort_skeleton_points(skeleton_points, start_from='bottom')

    fitter = CurveFitter(method='polynomial', poly_degree=2)
    curve_params, curve_func = fitter.fit_curve(sorted_points)

    if curve_params:
        print("Curve type: {}".format(curve_params['type']))
        if curve_params['type'] == 'polynomial':
            print("Polynomial coefficients: {}".format(curve_params['coefficients']))

    print("Example 4: Calculate Direction Vector")

    class MockCalib:
        def __init__(self):
            self.fx = 800.0
            self.fy = 800.0
            self.cx = 640.0
            self.cy = 360.0


    calib = MockCalib()

    calculator = MaskBasedDirectionCalculator(
        skeleton_method='distance_transform',
        curve_method='polynomial',
        poly_degree=2
    )

    direction_vector, target_point, confidence = calculator.calculate_direction_from_mask(
        path_mask,
        calib,
        depth_map=None,
        look_ahead_pixels=200
    )

    if direction_vector is not None:
        print("Direction vector: [{:.3f}, {:.3f}, {:.3f}]".format(*direction_vector))
        print("Target point: [{:.2f}, {:.2f}, {:.2f}]".format(*target_point))
        print("Confidence: {:.2f}".format(confidence))

        # Convert to steering angle
        angle_rad = np.arctan2(direction_vector[0], direction_vector[2])
        angle_deg = np.degrees(angle_rad)
        print("Steering angle: {:.1f} degrees".format(angle_deg))
    else:
        print("Failed to calculate direction vector")
