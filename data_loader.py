import random
from torch.utils.data.dataset import Dataset
import torchaudio
from utils import Genre
from torch.utils.data import DataLoader
import julius
import os
import json
from tqdm import tqdm


class Audioset:
    def __init__(self, files=None, shuffle=True, sample_rate=None, feature_cache=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.sample_rate = sample_rate
        self.feature_cache = feature_cache
        self.tqdm = tqdm(total=len(self.files))
        if shuffle:
            random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = self.files[index]
        file_path, genre = sample["path"], sample["label"]
        genre = genre.replace("-", "_")
        label = Genre[genre.upper()].value
        if self.tqdm.n < self.tqdm.total:
            self.tqdm.update(1)

        if self.feature_cache is None:
            out, sr = torchaudio.load(file_path)
        else:
            out = self.feature_cache[file_path]
        return out.squeeze(), label


class DataSet(Dataset):
    def __init__(self, json_dir, sample_rate=None, feature_cache=None):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param sample_rate: the signals sampling rate
        """
        files_json = os.path.join(json_dir)
        with open(files_json, 'r') as f:
            clean = json.load(f)

        self.dataset = Audioset(clean, sample_rate=sample_rate, feature_cache=feature_cache)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    train_json_path: str = "jsons/train.json"
    test_json_path: str = "jsons/test.json"

    dataset = DataSet(json_dir=train_json_path)
    loader = DataLoader(dataset, batch_size=32)
    for batch in loader:
        pass
    print("Done")
