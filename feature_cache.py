import torchaudio


class FeatureCache:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.cache = dict()

    def __getitem__(self, file_path):
        if file_path not in self.cache:
            self.cache[file_path] = self.feature_extractor.extract_normed_feats(torchaudio.load(file_path)[0])
        return self.cache[file_path]
