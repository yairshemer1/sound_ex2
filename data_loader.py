import random
from torch.utils.data.dataset import Dataset
import torchaudio
import math
import torch.nn.functional as F
from torch.utils.data.sampler import RandomSampler
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
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


def convert_audio(wav, from_samplerate, to_samplerate, channels):
    """Convert audio from a given samplerate to a target one and target number of channels."""
    wav = convert_audio_channels(wav, channels)
    return julius.resample_frac(wav, from_samplerate, to_samplerate)


class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 shuffle=True, sample_rate=None,
                 channels=None, convert=False):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.length = length
        self.stride = stride or length
        self.sample_rate = sample_rate
        self.channels = channels
        self.convert = convert
        if shuffle:
            random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = self.files[index]
        file_path, label = sample["path"], sample["label"]
        out, sr = torchaudio.load(file_path)
        return out, label


class DataSet(Dataset):
    def __init__(self, json_dir, length=None, stride=None,
                 sample_rate=None, convert=None):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param sample_rate: the signals sampling rate
        """
        files_json = os.path.join(json_dir)
        with open(files_json, 'r') as f:
            clean = json.load(f)

        kw = {'length': length, 'stride': stride, 'sample_rate': sample_rate, 'convert': convert}
        self.dataset = Audioset(clean, **kw)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    train_json_path: str = "jsons/train.json"
    test_json_path: str = "jsons/test.json"

    dataset = DataSet(json_dir=train_json_path, length=16000, stride=16000, sample_rate=16000, convert=True)
    loader = DataLoader(dataset, batch_size=32)
    for batch in loader:
        pass
    print("Done")
