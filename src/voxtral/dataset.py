from lhotse.cut import Cut
from transformers.utils import logging

from data.local_datasets import TS_ASR_Dataset, LhotseLongFormDataset

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


class TS_ASR_Dataset_1(TS_ASR_Dataset):
    def get_features(self, cut: Cut):
        samples, sr = cut.load_audio().squeeze(), cut.sampling_rate

        return samples, None

class LhotseLongFormDataset_1(LhotseLongFormDataset, TS_ASR_Dataset_1):
    pass