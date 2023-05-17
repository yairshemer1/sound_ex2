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
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

        self.mean = None
        self.std = None

    def extract_amplitude_envelope(self, one_wav):
        hop_length = 512
        amplitude_envelope = []
        for i in range(0, len(one_wav), hop_length):
            amplitude_envelope.append(np.abs(one_wav[i:i+hop_length]).max())
        return torch.tensor(amplitude_envelope)


    def extract_feats(self, one_wav):
        mfcc = librosa.feature.mfcc(y=one_wav.numpy(), sr=self.sample_rate).flatten()
        spectral_centroid = librosa.feature.spectral_centroid(y=one_wav.numpy(), sr=self.sample_rate).squeeze()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=one_wav.numpy()).squeeze()
        tempo = librosa.beat.tempo(y=one_wav.numpy(), sr=self.sample_rate).flatten()
        amplitude_envelope = self.extract_amplitude_envelope(one_wav.numpy().squeeze())
        amplitude_envelope_diff = np.diff(amplitude_envelope)



        # onset_env = librosa.onset.onset_strength(y=one_wav.numpy(), sr=self.sample_rate).squeeze()
        # chroma_cqt = librosa.feature.chroma_cqt(y=one_wav.numpy(), sr=self.sample_rate).squeeze()
        # chroma_cens = librosa.feature.chroma_cens(y=one_wav.numpy(), sr=self.sample_rate).squeeze()
        # chroma_stft = librosa.feature.chroma_stft(y=one_wav.numpy(), sr=self.sample_rate).squeeze()
        # spectral_contrast = librosa.feature.spectral_contrast(y=one_wav.numpy(), sr=self.sample_rate).squeeze()
        # spectral_bandwidth = librosa.feature.spectral_bandwidth(y=one_wav.numpy(), sr=self.sample_rate).squeeze()
        # spectral_rolloff = librosa.feature.spectral_rolloff(y=one_wav.numpy(), sr=self.sample_rate).squeeze()
        # spectral_flatness = librosa.feature.spectral_flatness(y=one_wav.numpy()).squeeze()

        return torch.Tensor(np.concatenate([mfcc, spectral_centroid, zero_crossing_rate, tempo, amplitude_envelope, amplitude_envelope_diff]))

    def normalize(self, x):
        # return x
        # return (x - self.mean) / np.maximum(self.std, 0.0001)
        return x - self.mean

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
