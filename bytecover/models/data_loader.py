import os
from typing import Dict, Literal, Tuple
import math

import h5py
import ffmpeg
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms

from bytecover.models.data_model import BatchDict
from bytecover.utils import bcolors


class ByteCoverDataset(IterableDataset):
    def __init__(
        self,
        data_path: str,
        file_ext: str,
        dataset_path: str,
        data_split: Literal["TRAIN", "VAL", "TEST"],
        debug: bool,
        target_sr: int,
        max_len: int,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.file_ext = file_ext
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.dataset_df = pd.read_csv("data/interim/shs100k.csv")
        self.debug = debug
        self.target_sr = target_sr
        self.max_len = max_len
        self._load_data()
        self.pipeline = transforms.Compose([self._read_cqt_20, self._pad_or_trim_audio_cqt])

    def __handle_workers(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process data loading
            start = 0
            end = len(self.dataset_df)
        else:
            # Multi-process data loading
            per_worker = int(math.ceil(len(self.dataset_df) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.dataset_df))
        return start, end
    
    def __iter__(self):
        start, end = self.__handle_workers()

        for index in range(start, end):
            track_id = self.track_ids[index]
            anchor_audio = self.pipeline(track_id)

            clique_id, pos_id, neg_id = self._triplet_sampling(track_id)

            if self.data_split == "TRAIN":
                positive_audio = self.pipeline(pos_id)
                negative_audio = self.pipeline(neg_id)
            else:
                positive_audio = torch.empty(0)
                negative_audio = torch.empty(0)
            
            yield dict(
                anchor_id=track_id,
                anchor=anchor_audio,
                anchor_label=torch.tensor(clique_id, dtype=torch.float),
                positive_id=pos_id,
                positive=positive_audio,
                negative_id=neg_id,
                negative=negative_audio,
            )

    def _triplet_sampling(self, track_id: str) -> Tuple[int, str, str]:
        clique_id = self.labels.loc[track_id, "clique"]
        versions = self.versions.loc[clique_id, "versions"]
        np.random.shuffle(versions)
        pos_list = np.setdiff1d(versions, track_id)
        pos_id = np.random.choice(pos_list, 1)[0]

        neg_id = self.labels[~self.labels.index.isin(versions)].sample(1).index[0]

        return (clique_id, pos_id, neg_id)

    def _load_data(self) -> None:
        self.track_ids = np.load(
            os.path.join(self.data_path, "splits", f"{self.data_split.lower()}_ids.npy"), allow_pickle=True
        )

        self.labels = pd.read_csv(os.path.join(self.data_path, "interim", "shs100k.csv"), usecols=["clique", "id"])
        self.labels = self.labels[self.labels["id"].isin(self.track_ids)]
        self.labels.dropna(inplace=True)
        self.labels.set_index("id", inplace=True)
        cliques = self.labels["clique"].unique()
        mapping = {}
        for k, clique in enumerate(cliques):
            mapping[clique] = k
        self.labels["clique"] = self.labels["clique"].map(lambda x: mapping[x])

        self.versions = pd.read_csv(
            os.path.join(self.data_path, "interim", "versions.csv"), converters={"versions": eval}
        )
        self.versions.dropna(inplace=True)
        self.versions = self.versions[self.versions["clique"].isin(cliques)]
        self.versions["clique"] = self.versions["clique"].map(lambda x: mapping[x])
        self.versions.set_index("clique", inplace=True)

    def _read_cqt_20(self, track_id: str) -> torch.Tensor:
        yt_id = self.dataset_df.loc[self.dataset_df.id == track_id,'Video ID'].iloc[0]
        fpath = os.path.join(self.dataset_path, str(ord(yt_id[0])), f"{yt_id}.{self.file_ext}") 

        try:
            with h5py.File(fpath, 'r') as f:
                audio_feature = f["cqt_20"][:]
        except FileNotFoundError:
            audio_feature = np.zeros((84, 300))
        return torch.from_numpy(audio_feature)
        
    def _read_audio(self, track_id: str) -> torch.Tensor:
        if self.debug:
            seq_len = np.random.randint(10, 200) if self.max_len <= 0 else self.max_len
            return torch.rand(seq_len * self.target_sr)
        filename = os.path.join(self.dataset_path, f"{track_id}.{self.file_ext}")

        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input(filename, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=self.target_sr)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(
                f"{bcolors.WARNING}Failed to load audio:{bcolors.FAIL + filename + bcolors.ENDC}\n{e.stderr.decode()}"
            ) from e

        # int16 ranges between -2^15 and +2^15 (±32768). By convention, floating point audio data is
        # normalized to the range of [-1.0, 1.0]
        audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

        return torch.from_numpy(audio)

    def _pad_or_trim_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if self.max_len <= 0:
            return audio

        if (self.data_split == "TRAIN") and (audio.shape[-1] <= self.max_len * self.target_sr):
            return F.pad(audio, (0, self.max_len * self.target_sr - audio.shape[-1]))

        max_offset = audio.shape[-1] - self.max_len * self.target_sr
        offset = np.random.randint(max_offset) if max_offset > 0 else 0

        return audio[offset : (offset + self.max_len * self.target_sr)]

    def _pad_or_trim_audio_cqt(self, audio: torch.Tensor) -> torch.Tensor:
                
        if self.max_len <= 0:
            return audio

        if (self.data_split == "TRAIN") and (audio.shape[-1] <= self.max_len):
            return F.pad(audio, (0, self.max_len - audio.shape[-1]))

        max_offset = audio.shape[-1] - self.max_len
        offset = np.random.randint(max_offset) if max_offset > 0 else 0

        return audio[:, offset:(offset + self.max_len)]

def bytecover_dataloader(
    data_path: str,
    file_ext: str,
    dataset_path: str,
    data_split: Literal["TRAIN", "VAL", "TEST"],
    debug: bool,
    max_len: int,
    batch_size: int,
    target_sr: int,
    **config: Dict,
) -> DataLoader:
    return DataLoader(
        ByteCoverDataset(data_path, file_ext, dataset_path, data_split, debug, target_sr=target_sr, max_len=max_len),
        batch_size=batch_size if max_len > 0 else 1,
        num_workers=config["num_workers"],
        # shuffle=config["shuffle"],
        drop_last=config["drop_last"],
    )
