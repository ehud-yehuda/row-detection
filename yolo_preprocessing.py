import mocks
import cv2
import numpy as np
from typing import Tuple, Union
import torch


class YOLOPreprocessor:
    def __init__(
            self,
            input_size: Tuple[int, int] = (640, 640),
            normalize: bool = True,
            letterbox: bool = True,
            auto_pad: bool = False,
            stride: int = 32
    ):
        self.input_size = input_size
        self.normalize = normalize
        self.letterbox = letterbox
        self.auto_pad = auto_pad
        self.stride = stride

    def load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def letterbox_resize(
            self,
            image: np.ndarray,
            new_shape: Tuple[int, int],
            color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:

        h, w = image.shape[:2]
        new_w, new_h = new_shape

        r = min(new_w / w, new_h / h)

        new_unpad_w = int(round(w * r))
        new_unpad_h = int(round(h * r))

        dw = new_w - new_unpad_w
        dh = new_h - new_unpad_h

        if self.auto_pad:
            dw = dw % self.stride
            dh = dh % self.stride

        dw /= 2
        dh /= 2

        if (w, h) != (new_unpad_w, new_unpad_h):
            image = cv2.resize(image, (new_unpad_w, new_unpad_h),
                               interpolation=cv2.INTER_LINEAR)

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color
        )

        return image, (r, r), (dw, dh)

    def simple_resize(
            self,
            image: np.ndarray,
            new_shape: Tuple[int, int]
    ) -> np.ndarray:
        return cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)

    def preprocess(
            self,
            image: Union[str, np.ndarray],
            return_original: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray, dict]]:
        if isinstance(image, str):
            original_image = self.load_image(image)
        else:
            original_image = image.copy()

        original_shape = original_image.shape[:2]

        if self.letterbox:
            processed_image, ratio, padding = self.letterbox_resize(
                original_image, self.input_size
            )
        else:
            processed_image = self.simple_resize(original_image, self.input_size)
            ratio = (self.input_size[0] / original_shape[1],
                     self.input_size[1] / original_shape[0])
            padding = (0, 0)

        if self.normalize:
            processed_image = processed_image.astype(np.float32) / 255.0

        processed_image = np.transpose(processed_image, (2, 0, 1))

        tensor = torch.from_numpy(processed_image).unsqueeze(0).float()

        if return_original:
            metadata = {
                'original_shape': original_shape,
                'ratio': ratio,
                'padding': padding,
                'input_size': self.input_size
            }
            return tensor, original_image, metadata

        return tensor

    def postprocess_coordinates(
            self,
            boxes: np.ndarray,
            metadata: dict
    ) -> np.ndarray:

        if len(boxes) == 0:
            return boxes

        ratio = metadata['ratio']
        padding = metadata['padding']

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]

        boxes[:, [0, 2]] /= ratio[0]
        boxes[:, [1, 3]] /= ratio[1]

        # Clip to original image bounds
        original_h, original_w = metadata['original_shape']
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h)

        return boxes



if __name__ == "__main__":
    preprocessor = YOLOPreprocessor(
        input_size=(640, 640),
        normalize=True,
        letterbox=True
    )
    print("Example 1: Single Image")

    dummy_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    tensor, original, metadata = preprocessor.preprocess(
        dummy_image,
        return_original=True
    )

    print(f"Original shape: {original.shape}")
    print(f"Preprocessed shape: {tensor.shape}")
    print(f"Metadata: {metadata}")

    print("Example 3: Postprocess Coordinates")

    model_boxes = np.array([
        [100, 150, 200, 250],  # x1, y1, x2, y2
        [300, 400, 400, 500]
    ], dtype=np.float32)

    print(f"Boxes in model space:\n{model_boxes}")

    original_boxes = preprocessor.postprocess_coordinates(model_boxes, metadata)

    print(f"Boxes in original space:\n{original_boxes}")