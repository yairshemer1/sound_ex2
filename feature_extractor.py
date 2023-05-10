import torchaudio


def extract_feats(wavs):
    torchaudio.transforms.MFCC()