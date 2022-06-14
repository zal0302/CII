import collections
from torchvision import transforms
from base import BaseDataLoader
from utils import transforms_pair
from .sod_dataset import SodDataset


class SodDataLoader(BaseDataLoader):
    """
    SOD data loading using BaseDataLoader
    """
    def __init__(self, data_dir, data_list, batch_size, image_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        assert isinstance(image_size, int) or (isinstance(image_size, collections.abc.Iterable) and len(image_size) == 2)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if training:
            trsfms = transforms_pair.ComposePair([
                # transforms_pair.RandomRotationPair(10),
                transforms_pair.RandomCropPair(),
                transforms_pair.ResizePair(image_size),
                # transforms_pair.RandomHorizontalFlipPair(),
                ])
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            target_trsfm = transforms.ToTensor()
        else:
            trsfms = transforms_pair.ComposePair([
                transforms_pair.ResizePair(image_size),
                transforms_pair.ToTensorPair(),
                ])
            trsfm = normalize
            target_trsfm = None

        self.data_dir = data_dir
        self.data_list = data_list
        self.dataset = SodDataset(self.data_dir, self.data_list, trsfms, trsfm, target_trsfm)
        # if training:
        #     super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, self.dataset._multiscale_collate)
        # else:
        #     super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)