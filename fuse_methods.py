import numpy as np


class VectorFusion:
    def __init__(
            self,
            fusion_method='weighted_average',
            min_confidence_threshold=0.3,
            angle_difference_threshold=45.0  # degrees
    ):
        self.fusion_method = fusion_method
        self.min_confidence_threshold = min_confidence_threshold
        self.angle_difference_threshold = angle_difference_threshold

    def fuse_vectors(
            self,
            vector1,
            confidence1,
            vector2,
            confidence2,
            method1_name="Method1",
            method2_name="Method2"
    ):
        valid1 = self._is_valid_vector(vector1, confidence1)
        valid2 = self._is_valid_vector(vector2, confidence2)

        fusion_info = {
            'method1_valid': valid1,
            'method2_valid': valid2,
            'method1_confidence': confidence1 if valid1 else 0.0,
            'method2_confidence': confidence2 if valid2 else 0.0,
            'fusion_method': self.fusion_method
        }

        if not valid1 and not valid2:
            return None, 0.0, fusion_info

        if valid1 and not valid2:
            fusion_info['used'] = method1_name
            return vector1, confidence1, fusion_info

        if not valid1 and valid2:
            fusion_info['used'] = method2_name
            return vector2, confidence2, fusion_info

        if self.fusion_method == 'weighted_average':
            return self._weighted_average_fusion(
                vector1, confidence1,
                vector2, confidence2,
                fusion_info
            )
        elif self.fusion_method == 'best_of':
            return self._best_of_fusion(
                vector1, confidence1,
                vector2, confidence2,
                fusion_info,
                method1_name, method2_name
            )
        elif self.fusion_method == 'validate':
            return self._validate_fusion(
                vector1, confidence1,
                vector2, confidence2,
                fusion_info,
                method1_name, method2_name
            )
        elif self.fusion_method == 'adaptive':
            return self._adaptive_fusion(
                vector1, confidence1,
                vector2, confidence2,
                fusion_info,
                method1_name, method2_name
            )
        else:
            raise ValueError("Unknown fusion method: {}".format(self.fusion_method))

    def _is_valid_vector(self, vector, confidence):
        if vector is None:
            return False
        if confidence < self.min_confidence_threshold:
            return False
        if not isinstance(vector, np.ndarray) or len(vector) != 3:
            return False
        if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            return False
        return True

    def _weighted_average_fusion(
            self,
            vector1, confidence1,
            vector2, confidence2,
            fusion_info
    ):
        total_conf = confidence1 + confidence2
        w1 = confidence1 / total_conf
        w2 = confidence2 / total_conf

        fused = w1 * vector1 + w2 * vector2

        norm = np.linalg.norm(fused)
        if norm < 1e-6:
            return None, 0.0, fusion_info

        fused = fused / norm

        fused_confidence = w1 * confidence1 + w2 * confidence2

        fusion_info['used'] = 'weighted_average'
        fusion_info['weight1'] = w1
        fusion_info['weight2'] = w2

        return fused, fused_confidence, fusion_info

    def _best_of_fusion(
            self,
            vector1, confidence1,
            vector2, confidence2,
            fusion_info,
            method1_name, method2_name
    ):
        if confidence1 >= confidence2:
            fusion_info['used'] = method1_name
            return vector1, confidence1, fusion_info
        else:
            fusion_info['used'] = method2_name
            return vector2, confidence2, fusion_info

    def _validate_fusion(
            self,
            vector1, confidence1,
            vector2, confidence2,
            fusion_info,
            method1_name, method2_name
    ):
        angle_diff = self._angle_between_vectors(vector1, vector2)

        fusion_info['angle_difference_deg'] = angle_diff

        if angle_diff < self.angle_difference_threshold:
            fused, conf, _ = self._weighted_average_fusion(
                vector1, confidence1,
                vector2, confidence2,
                {}
            )
            conf = min(conf * 1.2, 1.0)
            fusion_info['used'] = 'validated_average'
            fusion_info['agreement'] = True
            return fused, conf, fusion_info

        else:
            fusion_info['agreement'] = False
            if confidence1 >= confidence2:
                fusion_info['used'] = method1_name + '_fallback'
                return vector1, confidence1 * 0.7, fusion_info  # Penalty for disagreement
            else:
                fusion_info['used'] = method2_name + '_fallback'
                return vector2, confidence2 * 0.7, fusion_info

    def _adaptive_fusion(
            self,
            vector1, confidence1,
            vector2, confidence2,
            fusion_info,
            method1_name, method2_name
    ):
        angle_diff = self._angle_between_vectors(vector1, vector2)
        fusion_info['angle_difference_deg'] = angle_diff

        if angle_diff < 15.0:
            fused, conf, _ = self._weighted_average_fusion(
                vector1, confidence1,
                vector2, confidence2,
                {}
            )
            conf = min(conf * 1.3, 1.0)
            fusion_info['used'] = 'adaptive_close'
            return fused, conf, fusion_info

        elif angle_diff < self.angle_difference_threshold:
            fused, conf, _ = self._weighted_average_fusion(
                vector1, confidence1,
                vector2, confidence2,
                {}
            )
            conf = min(conf * 1.1, 1.0)
            fusion_info['used'] = 'adaptive_moderate'
            return fused, conf, fusion_info

        else:
            if confidence1 >= confidence2:
                fusion_info['used'] = 'adaptive_best_' + method1_name
                return vector1, confidence1 * 0.8, fusion_info
            else:
                fusion_info['used'] = 'adaptive_best_' + method2_name
                return vector2, confidence2 * 0.8, fusion_info

    def _angle_between_vectors(self, v1, v2):
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)

        angle_rad = np.arccos(cos_angle)

        angle_deg = np.degrees(angle_rad)

        return angle_deg


class ConfidenceCalculator:
    def __init__(self):
        pass

    def calculate_combined_confidence(
            self,
            method1_confidence,
            method2_confidence,
            fusion_info,
            num_trunks_detected=0,
            path_mask_coverage=0.0
    ):
        breakdown = {
            'method1_confidence': method1_confidence,
            'method2_confidence': method2_confidence,
            'fusion_confidence': fusion_info.get('fused_confidence', 0.0)
        }

        base_conf = (method1_confidence + method2_confidence) / 2.0

        if fusion_info.get('method1_valid') and fusion_info.get('method2_valid'):
            base_conf *= 1.2
            breakdown['both_methods_bonus'] = 0.2

        if fusion_info.get('agreement', False):
            base_conf *= 1.1
            breakdown['agreement_bonus'] = 0.1

        angle_diff = fusion_info.get('angle_difference_deg', 0)
        if angle_diff > 30:
            penalty = 1.0 - (angle_diff - 30) / 100.0  # Up to 0.4 penalty
            penalty = max(penalty, 0.6)
            base_conf *= penalty
            breakdown['angle_penalty'] = 1.0 - penalty

        if num_trunks_detected >= 6:
            base_conf *= 1.15
            breakdown['trunk_bonus'] = 0.15
        elif num_trunks_detected >= 4:
            base_conf *= 1.1
            breakdown['trunk_bonus'] = 0.1

        if path_mask_coverage > 0.2:
            base_conf *= 1.1
            breakdown['path_coverage_bonus'] = 0.1
        elif path_mask_coverage > 0.1:
            base_conf *= 1.05
            breakdown['path_coverage_bonus'] = 0.05

        overall_confidence = min(base_conf, 1.0)

        breakdown['overall_confidence'] = overall_confidence

        return overall_confidence, breakdown



if __name__ == "__main__":

    print("Example 1: Both Methods Valid - Close Agreement")

    vector1 = np.array([0.1, 0.0, 0.995])  # Slightly left
    confidence1 = 0.8

    vector2 = np.array([0.15, 0.0, 0.989])  # Also slightly left, similar
    confidence2 = 0.7

    fusion = VectorFusion(fusion_method='adaptive')

    fused, conf, info = fusion.fuse_vectors(
        vector1, confidence1,
        vector2, confidence2,
        "YOLO+RANSAC", "Segmentation"
    )

    if fused is not None:
        print("Method 1 vector: [{:.3f}, {:.3f}, {:.3f}], conf: {:.2f}".format(
            *vector1, confidence1
        ))
        print("Method 2 vector: [{:.3f}, {:.3f}, {:.3f}], conf: {:.2f}".format(
            *vector2, confidence2
        ))
        print("Fused vector: [{:.3f}, {:.3f}, {:.3f}]".format(*fused))
        print("Fused confidence: {:.2f}".format(conf))
        print("Fusion info: {}".format(info))


    print("Example 2: Methods Disagree")

    vector1 = np.array([0.1, 0.0, 0.995])  # Left
    confidence1 = 0.7

    vector2 = np.array([-0.2, 0.0, 0.98])  # Right (opposite!)
    confidence2 = 0.6

    fused, conf, info = fusion.fuse_vectors(
        vector1, confidence1,
        vector2, confidence2,
        "YOLO+RANSAC", "Segmentation"
    )

    if fused is not None:
        print("Method 1 vector: [{:.3f}, {:.3f}, {:.3f}], conf: {:.2f}".format(
            *vector1, confidence1
        ))
        print("Method 2 vector: [{:.3f}, {:.3f}, {:.3f}], conf: {:.2f}".format(
            *vector2, confidence2
        ))
        print("Fused vector: [{:.3f}, {:.3f}, {:.3f}]".format(*fused))
        print("Fused confidence: {:.2f}".format(conf))
        print("Angle difference: {:.1f} degrees".format(info.get('angle_difference_deg', 0)))
        print("Decision: {}".format(info.get('used')))


    print("Example 3: Only One Method Valid")

    vector1 = np.array([0.1, 0.0, 0.995])
    confidence1 = 0.8

    vector2 = None
    confidence2 = 0.1  # Too low

    fused, conf, info = fusion.fuse_vectors(
        vector1, confidence1,
        vector2, confidence2,
        "YOLO+RANSAC", "Segmentation"
    )

    if fused is not None:
        print("Method 1: Valid, conf: {:.2f}".format(confidence1))
        print("Method 2: Invalid, conf: {:.2f}".format(confidence2))
        print("Fused vector: [{:.3f}, {:.3f}, {:.3f}]".format(*fused))
        print("Fused confidence: {:.2f}".format(conf))
        print("Used: {}".format(info.get('used')))

    print("Example 4: Overall Confidence Calculation")

    conf_calc = ConfidenceCalculator()

    overall_conf, breakdown = conf_calc.calculate_combined_confidence(
        method1_confidence=0.8,
        method2_confidence=0.7,
        fusion_info={'method1_valid': True, 'method2_valid': True, 'agreement': True},
        num_trunks_detected=8,
        path_mask_coverage=0.25
    )

    print("Overall confidence: {:.2f}".format(overall_conf))
    for key, value in breakdown.items():
        print("  {}: {:.3f}".format(key, value))

    print("Example 5: Compare Fusion Methods")

    vector1 = np.array([0.1, 0.0, 0.995])
    confidence1 = 0.75
    vector2 = np.array([0.12, 0.0, 0.993])
    confidence2 = 0.7

    methods = ['weighted_average', 'best_of', 'validate', 'adaptive']

    for method in methods:
        fusion = VectorFusion(fusion_method=method)
        fused, conf, info = fusion.fuse_vectors(
            vector1, confidence1,
            vector2, confidence2
        )

        print("{}:".format(method))
        print("  Fused: [{:.3f}, {:.3f}, {:.3f}]".format(*fused))
        print("  Confidence: {:.2f}".format(conf))
        print("  Used: {}".format(info.get('used', 'N/A')))