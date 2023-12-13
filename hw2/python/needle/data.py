import numpy as np
import gzip
import struct
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            img = img[:, ::-1, :]
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        img_pad = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant', constant_values=0)
        img_crop = img_pad[self.padding + shift_x : self.padding + shift_x + img.shape[0], self.padding + shift_y : self.padding + shift_y + img.shape[1], :]
        return img_crop


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))
        self.batch_idx = 0

    def __iter__(self):
        if self.shuffle:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)),
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        else:
            self.ordering = np.array_split(np.arange(len(self.dataset)), 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        self.batch_idx = 0
        return self

    def __next__(self):
        if self.batch_idx >= len(self.ordering):
            raise StopIteration
        batch_indices = self.ordering[self.batch_idx]
        batch = self.dataset[batch_indices]
        self.batch_idx += 1
        return [Tensor(x) for x in batch]


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        # load the data - copied from hw1/apps/simple_ml.py::parse_mnist()
        with gzip.open(image_filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols, 1)
            X = X.astype(np.float32) / 255.0

        with gzip.open(label_filename, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            y = np.frombuffer(f.read(), dtype=np.uint8)

        self.images = X
        self.labels = y
        self.transforms = [] if transforms is None else transforms

    def __getitem__(self, index) -> object:
        image = self.images[index]
        label = self.labels[index]
        for func in self.transforms:
            image = func(image)
        return image, label

    def __len__(self) -> int:
        return len(self.labels)

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])