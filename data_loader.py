import random
from torch.utils.data.dataset import Dataset
import torchaudio
from utils import Genre
from torch.utils.data import DataLoader
import julius
import os
import json


def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError(
            'The audio file has less channels than requested but is not mono.'
        )
    return wav


def convert_audio(wav, from_samplerate, to_samplerate, channels):
    """Convert audio from a given samplerate to a target one and target number of channels."""
    wav = convert_audio_channels(wav, channels)
    return julius.resample_frac(wav, from_samplerate, to_samplerate)


class Audioset:
    def __init__(self, files=None, shuffle=True, sample_rate=None, convert=False):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.sample_rate = sample_rate
        self.convert = convert
        if shuffle:
            random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = self.files[index]
        file_path, genre = sample["path"], sample["label"]
        genre = genre.replace("-", "_")
        label = Genre[genre.upper()].value
        out, sr = torchaudio.load(file_path)
        return out.squeeze(), label


class DataSet(Dataset):
    def __init__(self, json_dir, sample_rate=None):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param sample_rate: the signals sampling rate
        """
        files_json = os.path.join(json_dir)
        with open(files_json, 'r') as f:
            clean = json.load(f)

        self.dataset = Audioset(clean, sample_rate=sample_rate)

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
