import numpy as np


class Line3D:
    def __init__(self, point, direction):
        self.point = point
        self.direction = direction

    def __repr__(self):
        return "Line3D(point={}, direction={})".format(
            self.point, self.direction
        )


class TrunkClusterer:
    def __init__(
            self,
            method='kmeans',
            path_direction='forward'
    ):
        self.method = method
        self.path_direction = path_direction

    def cluster_trunks(
            self,
            trunk_positions,
            tractor_position=None
    ):
        if len(trunk_positions) == 0:
            return np.array([]), np.array([]), np.array([])

        trunk_positions = np.array(trunk_positions)

        if tractor_position is None:
            tractor_position = np.zeros(3)

        if self.method == 'kmeans':
            labels = self._cluster_kmeans(trunk_positions)
        elif self.method == 'threshold':
            labels = self._cluster_threshold(trunk_positions, tractor_position)
        elif self.method == 'density':
            labels = self._cluster_density(trunk_positions)
        else:
            raise ValueError("Unknown clustering method: {}".format(self.method))

        left_trunks = trunk_positions[labels == 0]
        right_trunks = trunk_positions[labels == 1]

        return left_trunks, right_trunks, labels

    def _cluster_kmeans(self, positions):
        X = positions[:, 0].reshape(-1, 1)

        x_min, x_max = X.min(), X.max()
        c1 = x_min + 0.25 * (x_max - x_min)
        c2 = x_min + 0.75 * (x_max - x_min)
        centroids = np.array([c1, c2])

        for _ in range(10):
            distances = np.abs(X - centroids.reshape(1, -1))
            labels = np.argmin(distances, axis=1)

            for k in range(2):
                if np.sum(labels == k) > 0:
                    centroids[k] = np.mean(X[labels == k])

        if centroids[0] > centroids[1]:
            labels = 1 - labels

        return labels

    def _cluster_threshold(self, positions, tractor_position):
        X = positions[:, 0] - tractor_position[0]
        labels = (X > 0).astype(int)
        return labels

    def _cluster_density(self, positions):
        X = positions[:, 0]
        X_sorted = np.sort(X)

        gaps = np.diff(X_sorted)
        if len(gaps) == 0:
            return np.zeros(len(positions), dtype=int)

        max_gap_idx = np.argmax(gaps)
        threshold = (X_sorted[max_gap_idx] + X_sorted[max_gap_idx + 1]) / 2

        labels = (X > threshold).astype(int)
        return labels


class RANSACLineFitter:
    def __init__(
            self,
            max_iterations=1000,
            distance_threshold=0.3,
            min_inliers=3
    ):
        self.max_iterations = max_iterations
        self.distance_threshold = distance_threshold
        self.min_inliers = min_inliers

    def fit_line(self, points):
        if len(points) < 2:
            return None, np.array([]), np.array([])

        points = np.array(points)
        n_points = len(points)

        best_line = None
        best_inliers = []
        best_score = 0

        for iteration in range(self.max_iterations):
            if n_points < 2:
                break

            idx = np.random.choice(n_points, 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]

            direction = p2 - p1
            direction_norm = np.linalg.norm(direction)

            if direction_norm < 1e-6:
                continue

            direction = direction / direction_norm

            distances = self._point_to_line_distance(points, p1, direction)

            inlier_mask = distances < self.distance_threshold
            n_inliers = np.sum(inlier_mask)

            if n_inliers > best_score and n_inliers >= self.min_inliers:
                best_score = n_inliers
                best_inliers = inlier_mask
                best_line = Line3D(point=p1, direction=direction)

        if best_line is None:
            return self._fit_line_least_squares(points)

        inlier_points = points[best_inliers]
        refined_line, _, _ = self._fit_line_least_squares(inlier_points)

        return refined_line, inlier_points, best_inliers

    def _point_to_line_distance(self, points, line_point, line_direction):

        v = points - line_point

        projection_length = np.dot(v, line_direction)

        closest_points = line_point + projection_length[:, np.newaxis] * line_direction

        distances = np.linalg.norm(points - closest_points, axis=1)

        return distances

    def _fit_line_least_squares(self, points):
        if len(points) < 2:
            return None, points, np.ones(len(points), dtype=bool)
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        if len(points) >= 2:
            U, S, Vt = np.linalg.svd(centered)
            direction = Vt[0]

            if direction[2] < 0:
                direction = -direction

            line = Line3D(point=centroid, direction=direction)

            inlier_mask = np.ones(len(points), dtype=bool)

            return line, points, inlier_mask

        return None, np.array([]), np.array([])


class CenterLineCalculator:
    def calculate_center_line(
            self,
            left_line,
            right_line,
            method='midpoint'
    ):
        if left_line is None or right_line is None:
            return left_line if left_line is not None else right_line

        if method == 'midpoint':
            midpoint = (left_line.point + right_line.point) / 2.0

            avg_direction = left_line.direction + right_line.direction
            avg_direction = avg_direction / np.linalg.norm(avg_direction)

            center_line = Line3D(point=midpoint, direction=avg_direction)

        elif method == 'average_direction':
            avg_direction = (left_line.direction + right_line.direction) / 2.0
            avg_direction = avg_direction / np.linalg.norm(avg_direction)

            midpoint = (left_line.point + right_line.point) / 2.0

            center_line = Line3D(point=midpoint, direction=avg_direction)
        else:
            raise ValueError("Unknown method: {}".format(method))

        return center_line

    def calculate_direction_vector(
            self,
            center_line,
            tractor_position,
            look_ahead_distance=5.0
    ):

        if center_line is None:
            return None, None

        tractor_position = np.array(tractor_position)

        target_point = center_line.point + look_ahead_distance * center_line.direction

        direction_vector = target_point - tractor_position

        norm = np.linalg.norm(direction_vector)
        if norm < 1e-6:
            return None, target_point

        direction_vector = direction_vector / norm

        return direction_vector, target_point



if __name__ == "__main__":
    print("Geometric Processing - Clustering & Line Fitting")

    print("Example 1: Generate Mock Trunk Positions")

    np.random.seed(42)

    n_left = 8
    left_trunks_true = np.zeros((n_left, 3))
    left_trunks_true[:, 0] = -1.5 + np.random.randn(n_left) * 0.1  # X with noise
    left_trunks_true[:, 1] = 0.5 + np.random.randn(n_left) * 0.05  # Y (height)
    left_trunks_true[:, 2] = np.linspace(3, 18, n_left) + np.random.randn(n_left) * 0.2  # Z

    n_right = 7
    right_trunks_true = np.zeros((n_right, 3))
    right_trunks_true[:, 0] = 1.5 + np.random.randn(n_right) * 0.1
    right_trunks_true[:, 1] = 0.5 + np.random.randn(n_right) * 0.05
    right_trunks_true[:, 2] = np.linspace(4, 19, n_right) + np.random.randn(n_right) * 0.2

    all_trunks = np.vstack([left_trunks_true, right_trunks_true])

    print("Generated {} trunk positions".format(len(all_trunks)))
    print("  Left row: {} trunks".format(n_left))
    print("  Right row: {} trunks".format(n_right))

    print("Example 2: Cluster Trunks into Left/Right Rows")

    clusterer = TrunkClusterer(method='kmeans')
    left_trunks, right_trunks, labels = clusterer.cluster_trunks(all_trunks)

    print("Clustering results:")
    print("  Left cluster: {} trunks".format(len(left_trunks)))
    print("  Right cluster: {} trunks".format(len(right_trunks)))
    print("  Labels: {}".format(labels))

    fitter = RANSACLineFitter(
        max_iterations=1000,
        distance_threshold=0.3,
        min_inliers=3
    )

    left_line, left_inliers, left_mask = fitter.fit_line(left_trunks)
    right_line, right_inliers, right_mask = fitter.fit_line(right_trunks)

    if left_line:
        print("\nLeft line:")
        print("  Point: [{:.2f}, {:.2f}, {:.2f}]".format(*left_line.point))
        print("  Direction: [{:.3f}, {:.3f}, {:.3f}]".format(*left_line.direction))
        print("  Inliers: {} / {}".format(len(left_inliers), len(left_trunks)))

    if right_line:
        print("\nRight line:")
        print("  Point: [{:.2f}, {:.2f}, {:.2f}]".format(*right_line.point))
        print("  Direction: [{:.3f}, {:.3f}, {:.3f}]".format(*right_line.direction))
        print("  Inliers: {} / {}".format(len(right_inliers), len(right_trunks)))

    print("Example 4: Calculate Center Line")

    calculator = CenterLineCalculator()
    center_line = calculator.calculate_center_line(left_line, right_line)

    if center_line:
        print("Center line:")
        print("  Point: [{:.2f}, {:.2f}, {:.2f}]".format(*center_line.point))
        print("  Direction: [{:.3f}, {:.3f}, {:.3f}]".format(*center_line.direction))

    print("Example 5: Calculate Direction Vector for Tractor")

    tractor_position = np.array([0.0, 0.0, 0.0])

    direction_vector, target_point = calculator.calculate_direction_vector(
        center_line,
        tractor_position,
        look_ahead_distance=5.0
    )

    if direction_vector is not None:
        print("Tractor position: [{:.2f}, {:.2f}, {:.2f}]".format(*tractor_position))
        print("Target point (5m ahead): [{:.2f}, {:.2f}, {:.2f}]".format(*target_point))
        print("Direction vector: [{:.3f}, {:.3f}, {:.3f}]".format(*direction_vector))

        angle_rad = np.arctan2(direction_vector[0], direction_vector[2])
        angle_deg = np.degrees(angle_rad)
        print("Steering angle: {:.1f} degrees".format(angle_deg))

