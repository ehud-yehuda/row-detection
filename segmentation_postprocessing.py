import mocks
import cv2
import numpy as np
import torch
import sys
F = sys.modules['torch.nn.functional']


class SegmentationResult:
    def __init__(self, mask, confidence_map, class_masks, statistics):
        self.mask = mask
        self.confidence_map = confidence_map
        self.class_masks = class_masks
        self.statistics = statistics

    def get_class_mask(self, class_name):
        return self.class_masks.get(class_name, np.zeros_like(self.mask))

    def __repr__(self):
        return "SegmentationResult(shape={}, classes={})".format(
            self.mask.shape, list(self.class_masks.keys())
        )

class DeepLabV3Postprocessor:
    def __init__(
            self,
            class_names,
            confidence_threshold=0.5,
            min_area_threshold=100
    ):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.confidence_threshold = confidence_threshold
        self.min_area_threshold = min_area_threshold

    def process_output(
            self,
            output,
            metadata=None,
            apply_softmax=True
    ):
        output = output.squeeze(0)
        if apply_softmax:
            probs = F.softmax(output, dim=0)
        else:
            probs = output

        confidence_map, class_mask = torch.max(probs, dim=0)

        class_mask = class_mask.cpu().numpy()

        class_mask = class_mask .astype(np.uint8)

        confidence_map = confidence_map.cpu().numpy().astype(np.float32)

        low_confidence = confidence_map < self.confidence_threshold
        class_mask[low_confidence] = 0

        if metadata is not None and metadata.get('scale') is not None:
            class_mask = self._remove_padding_and_resize(class_mask, metadata)
            confidence_map = self._remove_padding_and_resize(confidence_map, metadata)

        class_masks = self._create_class_masks(class_mask)

        class_masks = self._filter_small_regions(class_masks)

        statistics = self._calculate_statistics(class_mask, confidence_map, class_masks)

        result = SegmentationResult(
            mask=class_mask,
            confidence_map=confidence_map,
            class_masks=class_masks,
            statistics=statistics
        )

        return result

    def _remove_padding_and_resize(
            self,
            mask,
            metadata
    ):
        x_offset, y_offset = metadata['offset']
        resized_h, resized_w = metadata['resized_shape']

        cropped = mask[y_offset:y_offset + resized_h, x_offset:x_offset + resized_w]

        print("  Cropped shape:", cropped.shape)

        orig_h, orig_w = metadata['original_shape']
        resized = cv2.resize(
            cropped,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST  # Use nearest for masks
        )

        return resized

    def _create_class_masks(self, class_mask):
        class_masks = {}

        for class_id, class_name in self.class_names.items():
            binary_mask = (class_mask == class_id).astype(np.uint8)
            class_masks[class_name] = binary_mask

        return class_masks

    def _filter_small_regions(
            self,
            class_masks
    ):
        filtered_masks = {}

        for class_name, mask in class_masks.items():
            if class_name == 'background':
                filtered_masks[class_name] = mask
                continue

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )

            filtered = np.zeros_like(mask)
            for i in range(1, num_labels):  # Skip background (0)
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= self.min_area_threshold:
                    filtered[labels == i] = 1

            filtered_masks[class_name] = filtered

        return filtered_masks

    def _calculate_statistics(
            self,
            class_mask,
            confidence_map,
            class_masks
    ):
        total_pixels = class_mask.size

        stats = {
            'total_pixels': int(total_pixels),
            'avg_confidence': float(np.mean(confidence_map)),
            'by_class': {}
        }

        for class_name, mask in class_masks.items():
            pixel_count = int(np.sum(mask))
            percentage = (pixel_count / float(total_pixels)) * 100

            if pixel_count > 0:
                class_confidence = confidence_map[mask > 0]
                avg_conf = float(np.mean(class_confidence))
            else:
                avg_conf = 0.0

            stats['by_class'][class_name] = {
                'pixel_count': pixel_count,
                'percentage': percentage,
                'avg_confidence': avg_conf
            }

        return stats

    def visualize_segmentation(
            self,
            image,
            result,
            alpha=0.5,
            colors=None
    ):
        if colors is None:
            colors = {
                'background': (0, 0, 0),
                'path': (0, 255, 0),  # Green
                'leaf': (0, 128, 0),  # Dark green
                'trunk': (139, 69, 19),  # Brown
                'pole': (128, 128, 128)  # Gray
            }

        h, w = result.mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_name, mask in result.class_masks.items():
            if class_name in colors:
                color = colors[class_name]
                colored_mask[mask > 0] = color

        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h))

        visualization = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

        return visualization


if __name__ == '__main__':
    class_names = {
        0: 'background',
        1: 'path',
        2: 'leaf',
        3: 'trunk'
    }

    # Initialize postprocessor
    postprocessor = DeepLabV3Postprocessor(
        class_names=class_names,
        confidence_threshold=0.5,
        min_area_threshold=100
    )

    print("Example 1: Postprocessing")

    num_classes = len(class_names)
    model_output = torch.randn(1, num_classes, 513, 513)

    print("Model output shape: {}".format(model_output.shape))
    metadata =  {'original_shape': (1080, 1920),
                 'scale': 0.2671875,
                 'offset': (0, 112),
                 'resized_shape': (288, 513),
                  'target_shape': (513, 513)}

    result = postprocessor.process_output(
        model_output,
        metadata=metadata,
        apply_softmax=True
    )


    print("Segmentation result:")
    print("  Mask shape: {}".format(result.mask.shape))
    print("  Confidence map shape: {}".format(result.confidence_map.shape))
    print("  Average confidence: {:.3f}".format(result.statistics['avg_confidence']))

    print("Example 2: Statistics by Class")

    for class_name, stats in result.statistics['by_class'].items():
        print("\n{}:".format(class_name.upper()))
        print("  Pixels: {:,}".format(stats['pixel_count']))
        print("  Percentage: {:.2f}%".format(stats['percentage']))
        print("  Avg confidence: {:.3f}".format(stats['avg_confidence']))


    print("Example 3: Extract Path Mask")

    path_mask = result.get_class_mask('path')
    print("\nPath mask shape: {}".format(path_mask.shape))
    print("Path pixels: {:,}".format(np.sum(path_mask)))
