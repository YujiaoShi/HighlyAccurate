import torch
import pytorch_lightning as pl

from . import get_dataset_module_by_name

# data_config: config/data/nuscenes.yaml
# loader_config: config/config.yaml
"""
loader:
  batch_size: 4
  num_workers: 4
  pin_memory: True
  prefetch_factor: 45
"""
class DataModule(pl.LightningDataModule):
    def __init__(self, dataset: str, data_config: dict, loader_config: dict):
        super().__init__()

        self.get_data = get_dataset_module_by_name(dataset).get_data

        self.data_config = data_config
        self.loader_config = loader_config

    def get_split(self, split, loader=True, shuffle=False):
        # get a list of NuScenesDataset
        datasets = self.get_data(split=split, **self.data_config)

        if not loader:
            return datasets

        # Concatenate a list of NuScenesDataset => one dataset
        dataset = torch.utils.data.ConcatDataset(datasets)

        loader_config = dict(self.loader_config)

        if loader_config['num_workers'] == 0:
            loader_config['prefetch_factor'] = 2
        # return a DataLoader as "train" or "val" dataloader
        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, **loader_config)

    def train_dataloader(self, shuffle=True):
        return self.get_split('train', loader=True, shuffle=shuffle)

    def val_dataloader(self, shuffle=True):
        return self.get_split('val', loader=True, shuffle=shuffle)
