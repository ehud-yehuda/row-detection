import mocks
import cv2
import numpy as np
import torch


class DeepLabV3Preprocessor:

    def __init__(
            self,
            input_size=(320, 240),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            maintain_aspect_ratio=True
    ):
        self.input_size = input_size
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)
        self.maintain_aspect_ratio = maintain_aspect_ratio

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image from {}".format(image_path))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def resize_with_padding(
            self,
            image,
            target_size,
            padding_color=(0, 0, 0)
    ):
        h, w = image.shape[:2]
        target_w, target_h = target_size

        scale = min(target_w / float(w), target_h / float(h))

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)

        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2

        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        metadata = {
            'original_shape': (h, w),
            'scale': scale,
            'offset': (x_offset, y_offset),
            'resized_shape': (new_h, new_w),
            'target_shape': target_size
        }

        return padded, metadata

    def simple_resize(self, image, target_size):
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    def normalize(self, image):
        image = image.astype(np.float32) / 255.0

        normalized = (image - self.mean) / self.std

        return normalized

    def preprocess(
            self,
            image,
            return_metadata=False
    ):
        if isinstance(image, str):
            original_image = self.load_image(image)
        else:
            original_image = image.copy()

        if self.maintain_aspect_ratio:
            processed_image, metadata = self.resize_with_padding(
                original_image, self.input_size
            )
        else:
            processed_image = self.simple_resize(original_image, self.input_size)
            metadata = {
                'original_shape': original_image.shape[:2],
                'target_shape': self.input_size,
                'scale': None,
                'offset': None
            }

        normalized = self.normalize(processed_image)

        tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()

        tensor = tensor.unsqueeze(0)

        if return_metadata:
            return tensor, original_image, metadata

        return tensor

if __name__ == "__main__":
    class_names = {
        0: 'background',
        1: 'path',
        2: 'leaf',
        3: 'trunk'
    }

    # Initialize preprocessor
    preprocessor = DeepLabV3Preprocessor(
        input_size=(513, 513),
        maintain_aspect_ratio=True
    )

    print("Example 1: Preprocessing")

    dummy_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    tensor, original, metadata = preprocessor.preprocess(
        dummy_image,
        return_metadata=True
    )

    print("Original shape: {}".format(original.shape))
    print("Preprocessed shape: {}".format(tensor.shape))
    print("Mean normalization: {}".format(preprocessor.mean.flatten()))
    print("Std normalization: {}".format(preprocessor.std.flatten()))
    print("Metadata: {}".format(metadata))
