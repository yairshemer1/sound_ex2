import os
import torch
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import torchaudio
from tqdm import tqdm


class FeatureExtractor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

        self.mean = None
        self.std = None

    def extract_amplitude_envelope(self, one_wav):
        hop_length = 512
        amplitude_envelope = []
        for i in range(0, len(one_wav), hop_length):
            amplitude_envelope.append(np.abs(one_wav[i : i + hop_length]).max())
        return torch.tensor(amplitude_envelope)

    def extract_feats(self, one_wav):
        mfcc = torchaudio.transforms.MFCC(sample_rate=self.sample_rate, n_mfcc=50)(one_wav).numpy().squeeze()
        one_wav = one_wav.numpy().squeeze()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=one_wav).squeeze()
        spectral_centroid = librosa.feature.spectral_centroid(y=one_wav, sr=self.sample_rate).squeeze()
        tempo = librosa.beat.tempo(y=one_wav, sr=self.sample_rate).flatten()
        chroma_stft = librosa.feature.chroma_stft(y=one_wav, sr=self.sample_rate).flatten()
        rms = librosa.feature.rms(y=one_wav).squeeze()
        amplitude_envelope = self.extract_amplitude_envelope(one_wav)
        amplitude_envelope_diff = np.diff(amplitude_envelope)

        stats_feats = [one_wav.std(), one_wav.mean()] + [np.percentile(one_wav, i) for i in range(0, 101, 10)]
        mfcc_stats = [mfcc.std(axis=1), mfcc.mean(axis=1)] + [np.percentile(mfcc, i, axis=1) for i in range(0, 101, 10)]
        return torch.Tensor(
            np.concatenate(
                [
                    np.array(mfcc_stats).flatten(),
                    np.array(stats_feats),
                    mfcc.flatten(),
                    chroma_stft,
                    zero_crossing_rate,
                    tempo,
                    rms,
                    amplitude_envelope,
                    amplitude_envelope_diff,
                    spectral_centroid,
                ]
            )
        )

    def normalize(self, x):
        return x
        # return (x - self.mean) / np.maximum(self.std, 0.0001)
        # return x - self.mean

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
        feats_arr = [self.extract_feats(wav) for wav, _ in tqdm(train_dataset)]
        feats_tensor = torch.stack(feats_arr)
        mean = feats_tensor.mean(dim=0)
        std = feats_tensor.std(dim=0)
        self.save_mean_std(save_dir, mean, std)


if __name__ == '__main__':
    out, sr = torchaudio.load("parsed_data/classical/train/1.mp3")
    feat_extractor = FeatureExtractor(sample_rate=sr)
    print(feat_extractor.extract_feats(out).shape)
