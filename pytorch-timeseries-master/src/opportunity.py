from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src import models
from src.trainer import BaseTrainer

from typing import Dict, List, Tuple, Optional
import os

config_info = {
    'epoch': 150,
    'lr': 0.01,
    'batch_size': 64,
    'momemtum': 0.9
}
# (557963, 113)
# (557963,)
# (118750, 113)
# (118750,)

# # This is for parsing the X data, you can ignore it if you do not need preprocessing
# def format_data_x(datafile):
#     x_data = None
#     for item in datafile:
#         item_data = np.loadtxt(item, dtype=np.double)
#         if x_data is None:
#             x_data = np.zeros((len(item_data), 1), dtype=np.double)
#         x_data = np.hstack((x_data, item_data))
#     x_data = x_data[:, 1:]
#     print(x_data.shape)
#     X = None
#     for i in range(len(x_data)):
#         row = np.asarray(x_data[i, :])
#         row = row.reshape(9, 128).T
#         if X is None:
#             X = np.zeros((len(x_data), 128, 9))
#         X[i] = row
#     print(X.shape)
#     return X
#
# # This is for parsing the Y data, you can ignore it if you do not need preprocessing
# def format_data_y(datafile):
#     data = np.loadtxt(datafile, dtype=np.int) - 1
#     YY = np.eye(6)[data]
#     return YY

# Load data function, if there exists parsed data file, then use it
# If not, parse the original dataset from scratch

# def load_data(data_folder):
#     if os.path.isfile(config_info['data_folder'] + 'data_har.npz'):
#         data = np.load(config_info['data_folder'] + 'data_har.npz')
#         X_train = data['X_train']
#         Y_train = data['Y_train']
#         X_test = data['X_test']
#         Y_test = data['Y_test']
#     else:
#         # This for processing the dataset from scratch
#         # After downloading the dataset, put it to somewhere that str_folder can find
#         str_folder = data_folder
#         INPUT_SIGNAL_TYPES = [
#             "body_acc_x_",
#             "body_acc_y_",
#             "body_acc_z_",
#             "body_gyro_x_",
#             "body_gyro_y_",
#             "body_gyro_z_",
#             "total_acc_x_",
#             "total_acc_y_",
#             "total_acc_z_"
#         ]
#         str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in
#                            INPUT_SIGNAL_TYPES]
#         str_test_files = [str_folder + 'test/' + 'Inertial Signals/' +
#                           item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
#         str_train_y = str_folder + 'train/y_train.txt'
#         str_test_y = str_folder + 'test/y_test.txt'
#         X_train = format_data_x(str_train_files)
#         X_test = format_data_x(str_test_files)
#         Y_train = format_data_y(str_train_y)
#         Y_test = format_data_y(str_test_y)
#     # return X_train, onehot_to_label(Y_train), X_test, onehot_to_label(Y_test)
#     return X_train, Y_train, X_test, Y_test

# def onehot_to_label(y_onehot):
#     a = np.argwhere(y_onehot == 1)
#     return a[:, -1]
def format_data_x(X):
    X_new = None
    for i in range(len(X)):
        row = np.asarray(X[i, :])
        row = row.reshape(113, 1).T
        if X_new is None:
            X_new = np.zeros((len(X), 1, 113))
        X_new[i] = row
    print(X_new.shape)
    return X_new

def load_opportunity():
    X_train = np.load('../data/Opp_X_train.npy')
    y_train = np.load('../data/Opp_y_train.npy') - 1
    X_test = np.load('../data/Opp_X_test.npy')
    y_test = np.load('../data/Opp_y_test.npy') - 1
    X_train = format_data_x(X_train)
    X_test = format_data_x(X_test)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    load_opportunity()

class data_loader(Dataset):
    def __init__(self, samples, labels, t):
        self.samples = torch.from_numpy(samples).float()
        self.labels = torch.from_numpy(labels).float()
        self.T = t
    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target
    def __len__(self):
        return len(self.samples)

# def normalize(x):
#     x_min = x.min(axis=(0, 2, 3), keepdims=True)
#     x_max = x.max(axis=(0, 2, 3), keepdims=True)
#     x_norm = (x - x_min) / (x_max - x_min)
#     return x_norm

def load(batch_size=64):
    x_train, y_train, x_test, y_test = load_opportunity()
    y_train = np.eye(17)[y_train]
    y_test = np.eye(17)[y_test]
    #x_train, x_test = x_train.reshape(
    #    (-1, 9, 1, 128)), x_test.reshape((-1, 9, 1, 128))
    x_train, x_test = x_train.reshape(
         (-1, 113, 1)), x_test.reshape((-1, 113, 1))
    transform = None # de aplicat encoding la y-uri
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
    print(x_train.shape)
    print(x_val.shape)
    train_set = data_loader(x_train, y_train, transform)
    val_set = data_loader(x_val, y_val, transform)
    test_set = data_loader(x_test, y_test, transform)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# if __name__ == '__main__':
#     train_loader, val_loader, test_loader = load(
#     batch_size=config_info['batch_size'])
#     print(len(train_loader.dataset))

class OPPTrainer(BaseTrainer):
    """Train the model on UCR datasets

    Attributes
    ----------
    model:
        The initialized inception model
    data_folder:
        The location of the data_folder
    """

    def __init__(self, model: nn.Module,
                 data_folder: Path = Path('data')) -> None:
        self.model = model

        self.data_folder = data_folder

        self.model_dir = data_folder / 'models' / self.model.__class__.__name__
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # to be filled by the fit function
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.test_results: Dict[str, float] = {}

        self.encoder: Optional[OneHotEncoder] = None

    # def _load_data(self) -> Tuple[InputData, InputData]:
    #     experiment_datapath = self.data_folder / 'UCR_TS_Archive_2015' / self.experiment
    #     if self.encoder is None:
    #         train, test, encoder = load_ucr_data(experiment_datapath)
    #         self.encoder = encoder
    #     else:
    #         train, test, _ = load_ucr_data(experiment_datapath, encoder=self.encoder)
    #     return train, test

    def get_loaders(self, batch_size: int, mode: str,
                    val_size: Optional[float] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Return dataloaders of the training / test data

        Arguments
        ----------
        batch_size:
            The batch size each iteration of the dataloader should return
        mode: {'train', 'test'}
            If 'train', this function should return (train_loader, val_loader)
            If 'test', it should return (test_loader, None)
        val_size:
            If mode == 'train', the fraction of training data to use for validation
            Ignored if mode == 'test'

        Returns
        ----------
        Tuple of (train_loader, val_loader) if mode == 'train'
        Tuple of (test_loader, None) if mode == 'test'
        """
        #train_data, test_data = self._load_data()
        train_loader, val_loader, test_loader = load(
             batch_size=config_info['batch_size'])
        # print(len(train_loader.dataset))

        if mode == 'train':
            # assert val_size is not None, 'Val size must be defined when loading training data'
            # train_data, val_data = train_data.split(val_size)
            #
            # train_loader = DataLoader(
            #     TensorDataset(train_data.x, train_data.y),
            #     batch_size=batch_size,
            #     shuffle=True,
            # )
            # val_loader = DataLoader(
            #     TensorDataset(val_data.x, val_data.y),
            #     batch_size=batch_size,
            #     shuffle=False
            # )

            return train_loader, val_loader
        else:
            # test_loader = DataLoader(
            #     TensorDataset(test_data.x, test_data.y),
            #     batch_size=batch_size,
            #     shuffle=False,
            # )
            return test_loader, None

    def save_model(self, savepath: Optional[Path] = None) -> Path:
        save_dict = {
            'model': {
                'model_class': self.model.__class__.__name__,
                'state_dict': self.model.state_dict(),
                'input_args': self.model.input_args,
            },
            'encoder': self.encoder
        }
        if savepath is None:
            model_name = f'{self.model.__class__.__name__}_model.pkl'
            savepath = self.model_dir / model_name
        torch.save(save_dict, savepath)

        return savepath


def load_opp_trainer(model_path: Path) -> OPPTrainer:

    data_folder = model_path.resolve().parents[2]

    model_dict = torch.load(model_path)

    model_class = getattr(models, model_dict['model']['model_class'])
    model = model_class(**model_dict['model']['input_args'])
    model.load_state_dict(model_dict['model']['state_dict'])

    loaded_trainer = OPPTrainer(model,
                                data_folder=data_folder)
    loaded_trainer.encoder = model_dict['encoder']

    return loaded_trainer