import os.path

# Python libraries
import yaml
from os.path import join as opj
import torch as pt
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision.transforms as trafo

from typing import Dict


# Project imports
from Data.utils import std_norm_data_pt, minmax_norm_data_pt
from Data.datasets import *


class MedicalDataLoaderBase(pl.LightningDataModule):
    def __init__(
            self,
            root_dir: str,
            exp_path: str,
            batch_size: int,
            transforms: Dict[str, list] = None,
            rescale: bool = False,
            num_workers: int = 0,
            prefetch_factor: int = None,
            persistent_workers: bool = None
    ):
        super().__init__()
        assert os.path.exists(root_dir)

        self.root_dir = root_dir
        self.exp_path = exp_path
        self.batch_size = batch_size
        self.transforms = transforms
        self.rescale = rescale
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        with open(opj(self.exp_path, 'config.yaml'), 'r+') as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        # Set in setup()
        self.train_set = None
        self.test_set = None
        self.val_set = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers,
                          pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers,
                          pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers,
                          pin_memory=True)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers,
                          pin_memory=True)

    def _rescale_data(self, data: pt.Tensor, strategy: str) -> pt.Tensor:
        if strategy == "std_norm":
            return std_norm_data_pt(data, dim=(1, 2, 3))
        elif strategy == "unit_norm" or strategy == "minmax_norm":
            if strategy == "minmax_norm":
                norm_range = (-1., 1.)
            else:
                norm_range = (0., 1.)
            return minmax_norm_data_pt(data, norm_range=norm_range, dim=(1, 2, 3)) 
        
    def _resize_data(self, data: pt.Tensor, strategy: int) -> pt.Tensor:
        data = data.permute(0,3,1,2)
        b,c,h,w = data.shape
        
        # for some interesting reason F.pad does not work like it is supposed, that is F.pad(data, (0,1,0,4)) should pad the last dim with (0,1), and the second to last with (0,4), but here it is reversed,         (0,1) is for the second to last
        transforms = trafo.Compose([
            #trafo.Lambda(lambda x: x[:,:,1:,1:]),
            #trafo.Lambda(lambda x: F.pad(x, (0, self.config['fmri_settings']['resize'][0] - (h-1), self.config['fmri_settings']['resize'][1] - (w-1), 0))),
            trafo.Resize(size = (self.config['fmri_settings']['resize'][0] * 2, self.config['fmri_settings']['resize'][1] * 2))
        ])
        
        data = transforms(data)
        return data


class ABIDELoader(MedicalDataLoaderBase):
    def __init__(
            self,
            root_dir: str,
            exp_path: str,
            batch_size: int,
            transforms: Dict[str, list] = None,
            rescale: bool = False,
            num_workers: int = 0,
            prefetch_factor: int = None,
            persistent_workers: bool = None,
    ):
        """
        Handler class for the  ABIDE I pre-processed fMRI dataset.

        Args:
            root_dir: path of the directory containing data as a list of .nii.gz files
            exp_path: path of the directory containing subset configuration files in .csv format
            transforms: list of transformations as partial functions to be applied on the data
            rescale: whether to rescale (e.g. normalize, standardize) the data
        """
        super().__init__(root_dir, exp_path, batch_size, transforms, rescale,
                         num_workers, prefetch_factor, persistent_workers)

    def setup(self, stage: str, test_filename = None, load_video = False) -> None:
        dataset = ABIDEDataset if not load_video else ABIDEDatasetVideo
        # Load data
        if stage == 'fit':
            self.train_set = dataset(self.root_dir, opj(self.exp_path, 'train.csv'), self.config['rescale'], self.config['modalities'])
            self.val_set = dataset(self.root_dir, opj(self.exp_path, 'val.csv'), self.config['rescale'], self.config['modalities'])
        elif stage == 'test':
            self.test_set = dataset(self.root_dir, opj(self.exp_path, 'test.csv' if test_filename is None else test_filename), self.config['rescale'], self.config['modalities'])
        elif stage == 'pred4train':
            self.test_set = dataset(self.root_dir, opj(self.exp_path, 'train.csv'), self.config['rescale'], self.config['modalities'])
        elif stage == 'predict':
            print(self.test_set.set_file)
        elif stage == 'NYU_UM1':
            self.nyu_train = dataset(self.root_dir, opj(self.exp_path, 'NYU_train.csv'), self.config['rescale'], self.config['modalities'])
            self.nyu_test = dataset(self.root_dir, opj(self.exp_path, 'NYU_test.csv'), self.config['rescale'], self.config['modalities'])
            self.um1_train = dataset(self.root_dir, opj(self.exp_path, 'UM1_train.csv'), self.config['rescale'], self.config['modalities'])
            self.um1_test = dataset(self.root_dir, opj(self.exp_path, 'UM1_test.csv'), self.config['rescale'], self.config['modalities'])
        else:
            raise AttributeError(f'Wrong stage was given! ({stage}) Possible: fit, test, pred4train')

    def load_sites(self, site1, site2) -> None:
        self.tr1 = ABIDEDatasetVideo(self.root_dir, opj(self.exp_path, f'{site1}_train.csv'), self.config['rescale'], self.config['modalities'])
        self.ts1 = ABIDEDatasetVideo(self.root_dir, opj(self.exp_path, f'{site1}_test.csv'), self.config['rescale'], self.config['modalities'])
        self.tr2 = ABIDEDatasetVideo(self.root_dir, opj(self.exp_path, f'{site2}_train.csv'), self.config['rescale'], self.config['modalities'])
        self.ts2 = ABIDEDatasetVideo(self.root_dir, opj(self.exp_path, f'{site2}_test.csv'), self.config['rescale'], self.config['modalities'])

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        if self.transforms is not None:
            for key in self.transforms.keys():
                for transform in self.transforms[key]:
                    batch[key] = transform(batch[key])
                batch[key] = batch[key].contiguous()
        return batch
