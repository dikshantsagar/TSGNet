import lightning.pytorch as pl
from datasets.abberated_dataset import ElectronAbTrainDataset, ElectronAbTestDataset
from torch.utils.data import DataLoader, DistributedSampler

class AbberatedDataModule(pl.LightningDataModule):
    def __init__(self, directory, downsample_rate, batch_size=16, shuffle=True, pin_memory=True, num_workers=8):
        super().__init__()
        self.directory = directory
        self.downsample_rate = downsample_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = ElectronAbTrainDataset(self.directory, self.downsample_rate)
        self.test_dataset = ElectronAbTestDataset(self.directory, self.downsample_rate)
    
    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=self.shuffle)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            drop_last=True
        )

    def val_dataloader(self):
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            drop_last=True
        )