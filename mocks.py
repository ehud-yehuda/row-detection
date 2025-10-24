import sys
from unittest.mock import MagicMock
import numpy as np


class MockTensor:
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def __repr__(self):
        return "MockTensor({})".format(self.data.shape)

    def squeeze(self, dim=None):
        return mock_squeeze(self.data, dim)


def mock_torch_max(tensor, dim=None):
    if isinstance(tensor, MockTensor):
        data = tensor.data
    else:
        data = tensor

    if dim is None:
        return np.max(data)
    else:
        values = np.max(data, axis=dim).astype(np.float32)
        indices = np.argmax(data, axis=dim).astype(np.int64)
        return MockTensor(values), MockTensor(indices)


def mock_squeeze(tensor, dim=None):
    if isinstance(tensor, MockTensor):
        data = tensor.data
    else:
        data = tensor

    result = np.squeeze(data, axis=dim)
    if isinstance(result, np.ndarray) and result.ndim > 0:
        return MockTensor(result)
    return result


def mock_softmax(x, dim=0):
    print("\n*** CALLED mock_softmax ***")
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    result = e_x / np.sum(e_x, axis=dim, keepdims=True)
    return MockTensor(result)


class F:
    softmax = staticmethod(mock_softmax)

class MockTorchNN:
    Module = object
    functional = F

mock_torch = MagicMock()
mock_torch.tensor = lambda x, **_: MockTensor(np.array(x))
mock_torch.from_numpy = lambda x: MockTensor(x)
mock_torch.Tensor = MockTensor
mock_torch_nn = MockTorchNN()
mock_torch.randn = lambda *shape: np.random.randn(*shape).astype(np.float32)
mock_torch.zeros = lambda *shape: np.zeros(shape, dtype=np.float32)
mock_torch.ones = lambda *shape: np.ones(shape, dtype=np.float32)
mock_torch.from_numpy = lambda x: x
mock_torch.cat = lambda tensors, dim=0: np.concatenate(tensors, axis=dim)
mock_torch.squeeze = mock_squeeze
mock_torch.max = mock_torch_max

mock_torch_nn.functional = F

sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch_nn
sys.modules['torch.nn.functional'] = F