from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pickle
import os
import pandas as pd


class StockData(Dataset):
    """
    Stock Dataset containing earning calls transcripts and audio.
    """

    def __init__(
        self, df_path, embed, audio_dir, max_len, text_embed_dim, audio_embed_dim
    ):
        """
        df_path : path of the csv file containing the data
        embed: dict containing the sentence embeddings
        audio_dir: directory containing the audio files
        max_len: maximum number of sentences to use
        text_embed_dim: size of text embedding
        audio_embed_dim: size of audio embedding
        """
        self.df = pd.read_csv(df_path)
        self.keys = list(self.df["filename"].values)
        self.df.set_index("filename", inplace=True)
        self.embed = embed
        self.audio_dir = audio_dir
        self.max_len = max_len
        self.text_embed_dim = text_embed_dim
        self.audio_embed_dim = audio_embed_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data = torch.tensor(self.df.loc[key].values)
        y_true = data[-1]
        data = data[:-1]

        sentence = torch.zeros(size=(self.max_len, self.text_embed_dim))
        audio = torch.zeros(size=(self.max_len, self.audio_embed_dim))

        sentence_ = torch.tensor(self.embed[key])
        seq_len = sentence_.shape[0]

        audio_ = pd.read_csv(os.path.join(self.audio_dir, key, "features.csv"))
        audio_ = audio_.replace("--undefined-- ", np.nan)
        audio_ = audio_.replace("--undefined--", np.nan)
        audio_ = audio_.replace("--undefined-", np.nan)
        audio_ = audio_.ffill().values.astype(np.float)
        audio_ = np.nan_to_num(audio_)
        audio_ = torch.tensor(audio_)

        if seq_len > self.max_len:
            sentence_ = sentence_[: self.max_len]
            audio_ = audio_[: self.max_len]
            seq_len = self.max_len
        sentence[:seq_len, :] = sentence_
        audio[:seq_len, :] = audio_
        return data, y_true, sentence, audio, seq_len
