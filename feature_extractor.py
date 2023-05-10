import os
import torch
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from functools import partial

import torchaudio


class FeatureExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=40, n_fft=400, hop_length=160, n_mels=40):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.mean = None
        self.std = None

        self.mfcc_gen = partial(
            librosa.feature.mfcc,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

    # def extract_feats(self, wav):
    #     return torch.stack([self.extract_feats_one_example(one_wav) for one_wav in wav])

    def extract_feats(self, one_wav):
        # mfcc = self.mfcc_gen(y=one_wav.numpy())
        # return self.plot_log_mel_spectrogram(mfcc)

        spectral_centroid = librosa.feature.spectral_centroid(y=one_wav.numpy(), sr=self.sample_rate).squeeze()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=one_wav.numpy()).squeeze()
        return torch.Tensor(np.concatenate([spectral_centroid, zero_crossing_rate]))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def extract_normed_feats(self, wav):
        assert self.mean is not None and self.std is not None, "Mean and std not set"
        feats = self.extract_feats(wav)
        return self.normalize(feats)

    @staticmethod
    def plot_log_mel_spectrogram(mfcc):
        fig, ax = plt.subplots(figsize=(12, 5))
        img = librosa.display.specshow(librosa.power_to_db(mfcc, ref=np.max), y_axis='mel', x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()

    def save_mean_std(self, save_dir, mean, std):
        self.mean = mean
        self.std = std
        np.save(os.path.join(save_dir, "mean.npy"), mean)
        np.save(os.path.join(save_dir, "std.npy"), std)

    def load_mean_std(self, load_dir):
        self.mean = np.load(os.path.join(load_dir, "mean.npy"))
        self.std = np.load(os.path.join(load_dir, "std.npy"))

    def calc_mean_std(self, train_dataset, save_dir):
        feats_arr = [self.extract_feats(wav) for wav, _ in train_dataset]
        feats_tensor = torch.stack(feats_arr)
        mean = feats_tensor.mean(dim=0)
        std = feats_tensor.std(dim=0)
        self.save_mean_std(save_dir, mean, std)

if __name__ == '__main__':
    out, sr = torchaudio.load("parsed_data/classical/train/1.mp3")
    feat_extractor = FeatureExtractor(sample_rate=sr)
    print(feat_extractor.extract_feats(out).shape)
