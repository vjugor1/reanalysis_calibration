import os
import random
import pickle
from click import pass_context
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import torch


class WindDataModule(pl.LightningDataModule):
    def __init__(
        self, train_dirs: str, val_dirs: str, test_dirs: str, batch_size: int = 128
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_dirs, self.val_dirs, self.test_dirs = train_dirs, val_dirs, test_dirs
        # self.X_train, self.X_val, self.X_test = (
        #     torch.tensor(X["Train"], dtype=torch.double),
        #     torch.tensor(X["Val"], dtype=torch.double),
        #     torch.tensor(X["Test"], dtype=torch.double),
        # )
        # self.y_train, self.y_val, self.y_test = (
        #     torch.tensor(y["Train"], dtype=torch.double),
        #     torch.tensor(y["Val"], dtype=torch.double),
        #     torch.tensor(y["Test"], dtype=torch.double),
        # )
        # mean_channels = self.X_train.mean(dim=[0, -1, -2])
        # std_channels = self.X_train.std(dim=[0, -1, -2])
        # self.transform = transforms.Compose(
        #     [
        #         transforms.Normalize(mean=mean_channels, std=std_channels),
        #     ]
        # )

        self.dl_dict = {"batch_size": self.batch_size}

    def prepare_data(self):
        try:
            mean_channels = self.X_train.mean(dim=[0, -1, -2])
            std_channels = self.X_train.std(dim=[0, -1, -2])
            self.transform = transforms.Compose(
                [
                    transforms.Normalize(mean=mean_channels, std=std_channels),
                ]
            )
        except AttributeError:
            # train
            Xs, ys = [], []
            for station_dir in self.train_dirs:
                X_t, y_t = extract_station(station_dir)
                Xs.append(X_t)
                ys.append(y_t)
            self.X_train = torch.cat(Xs, dim=0).double()
            self.y_train = torch.cat(ys, dim=0).double()
            # val
            Xs, ys = [], []
            for station_dir in self.val_dirs:
                X_t, y_t = extract_station(station_dir)
                Xs.append(X_t)
                ys.append(y_t)
            self.X_val = torch.cat(Xs, dim=0).double()
            self.y_val = torch.cat(ys, dim=0).double()
            # test
            Xs, ys = [], []
            for station_dir in self.test_dirs:
                X_t, y_t = extract_station(station_dir)
                Xs.append(X_t)
                ys.append(y_t)
            self.X_test = torch.cat(Xs, dim=0).double()
            self.y_test = torch.cat(ys, dim=0).double()

            # normalize data
            mean_channels = self.X_train.mean(dim=[0, -1, -2])
            std_channels = self.X_train.std(dim=[0, -1, -2])
            self.transform = transforms.Compose(
                [
                    transforms.Normalize(mean=mean_channels, std=std_channels),
                ]
            )
        # if type(self.y_train) == torch.Tensor and len(self.y_train.shape) == 2:
        #     pass
        # else:

        #     self.y_train = torch.tensor(self.y_train, dtype=torch.long)
        #     self.y_val = torch.tensor(self.y_val, dtype=torch.long)
        #     self.y_test = torch.tensor(self.y_test, dtype=torch.long)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = TensorDataset(
                self.transform(self.X_train), torch.tensor(self.y_train)
            )
            self.dataset_val = TensorDataset(
                self.transform(self.X_val), torch.tensor(self.y_val)
            )

        if stage == "test" or stage is None:
            self.dataset_test = TensorDataset(
                self.transform(self.X_test), torch.tensor(self.y_test)
            )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, **self.dl_dict)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, **self.dl_dict)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, **self.dl_dict)


def train_val_test_split(
    path_to_data: str,
    train: float = 0.5,
    val: float = 0.25,
    test: float = 0.25,
    verbose: bool = False,
) -> dict:
    """randomly splits weather stations to train, val, test in proportions given

    Args:
        path_to_data (str): path to folder which contains folder with preparsed numpy objects from stations
        train (float, optional): train weather stations share. Defaults to 0.5.
        val (float, optional): val weather stations share. Defaults to 0.25.
        test (float, optional): test weather stations share. Defaults to 0.25.
        verbose (bool, optional): if to print out the result of split. Defaults to False.

    Returns:
        dict: [description]
    """
    stations = os.listdir(path_to_data)
    random.shuffle(stations)
    partition = {"train_share": train, "val_share": val, "test_share": test}
    train_len = int(len(stations) * partition["train_share"])
    val_len = int(len(stations) * partition["val_share"])
    test_len = int(len(stations) * partition["test_share"])
    train_sts, val_sts, test_sts = (
        stations[:train_len],
        stations[train_len : train_len + val_len],
        stations[train_len + val_len :],
    )
    st_split_dict = {"Train": train_sts, "Val": val_sts, "Test": test_sts}
    assert any(
        item not in st_split_dict["Train"] for item in st_split_dict["Val"]
    ), "Data leak: val in train"
    assert any(
        item not in st_split_dict["Train"] for item in st_split_dict["Test"]
    ), "Data leak: test in train"
    if verbose:
        print(st_split_dict)
    return st_split_dict


def extract_station(station_path: str):
    X = pd.read_csv(os.path.join(station_path, "objects.csv"))
    y = pd.read_csv(os.path.join(station_path, "targets.csv"))
    lons, lats = X.longitude.unique(), X.latitude.unique()
    lons.sort()
    lats.sort()
    timestamps = X.id.unique()
    timestamps.sort()
    n_channels = len(X.columns) - 3
    X_t = torch.zeros((len(timestamps), n_channels, len(lons), len(lats)))

    for lon_idx in range(len(lons)):
        for lat_idx in range(len(lats)):
            curr_pix_ts = (
                X[(X["longitude"] == lons[lon_idx]) & (X["latitude"] == lats[lat_idx])]
                .sort_values("id")
                .drop(columns=["id", "latitude", "longitude"])
            )
            for i, col in enumerate(sorted(curr_pix_ts.columns)):
                X_t[:, i, lon_idx, lat_idx] = torch.tensor(curr_pix_ts[col].values)
    y_t = torch.tensor(y["Средняя скорость ветра"].values)

    return X_t, y_t


def extract_splitted_data(path_to_dump: str, st_split_dict: dict) -> tuple:
    """extracts X, y, splitted into train, val, test

    Args:
        path_to_dump (str): path to folder which contains folder with preparsed numpy objects from stations
        st_split_dict (dict): division by stations' names into train, val, test

    Returns:
        tuple: (X - keys = train, val, test. values = objects; y - similarly)
    """
    X = {}
    y = {}
    for split_part, sts in st_split_dict.items():
        X_split = []
        y_split = []
        for st in sts:
            st_dir = os.path.join(path_to_dump, st)
            with open(os.path.join(st_dir, "objects.npy"), "rb") as f:
                # X_ = pickle.load(f)
                X_ = np.load(f)

            X_split.append(X_)
            try:
                with open(os.path.join(st_dir, "target.npy"), "rb") as f:
                    # y_ = pickle.load(f)
                    y_ = np.load(f)
                y_split.append(y_)
            except FileNotFoundError:
                y_split.append([])

        X[split_part] = np.concatenate(X_split)
        y[split_part] = np.concatenate(y_split)
    return X, y
