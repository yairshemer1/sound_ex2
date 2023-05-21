import torchaudio


class FeatureCache:
    def __init__(self, model):
        self.model = model
        self.cache = dict()

    def __getitem__(self, file_path):
        if file_path not in self.cache:
            self.cache[file_path] = self.model.extract_feats_one_wav(torchaudio.load(file_path)[0])
        return self.cache[file_path]
