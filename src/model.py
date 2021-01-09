import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VStock(nn.Module):
    """
    Forecasting model using LSTM
    """

    def __init__(
        self,
        text_embed_dim,
        audio_embed_dim,
        text_hidden_size,
        audio_hidden_size,
        use_stock_data=False,
    ):
        """
        text_embed_dim: size of sentence embeddings,
        audio_embed_dim: size of audio embeddings,
        text_hidden_size: hidden size to use inside text BLSTM,
        audio_hidden_size: hiden size to use inside audio BLSTM,
        use_stock_data: use stock data or not
        """
        super(VStock, self).__init__()
        self.tlstm = nn.LSTM(
            input_size=text_embed_dim,
            hidden_size=text_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.alstm = nn.LSTM(
            input_size=audio_embed_dim,
            hidden_size=audio_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.drop = nn.Dropout(p=0.45)

        self.linear1 = nn.Linear(text_hidden_size * 2, 1)
        self.linear2 = nn.Linear(audio_hidden_size * 2, 1)
        if use_stock_data:
            self.linear3 = nn.Linear(
                text_hidden_size * 2 + audio_hidden_size * 2 + 20, 128
            )
        else:
            self.linear3 = nn.Linear(text_hidden_size * 2 + audio_hidden_size * 2, 128)
        self.linear4 = nn.Linear(128, 1)

        self.batchnorm = nn.BatchNorm1d(128)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.use_stock_data = use_stock_data

    def forward(self, stock_feats, sentence_feat, audio_feats, sentence_len=None):
        """
        stock_feats: previous stock features
        sentence_feat: sentence features
        audio_feats : audio features

        """
        # text_packed_input = pack_padded_sequence(
        #     sentence_feat, sentence_len, batch_first=True, enforce_sorted=False
        # )
        # text_packed_output, _ = self.tlstm(text_packed_input)
        # text_output, _ = pad_packed_sequence(text_packed_output, batch_first=True)
        # print("text: ", text_output)
        # audio_packed_input = pack_padded_sequence(
        #     audio_feats, sentence_len, batch_first=True, enforce_sorted=False
        # )
        # audio_packed_output, _ = self.alstm(audio_packed_input)
        # audio_output, _ = pad_packed_sequence(audio_packed_output, batch_first=True)

        text_output, _ = self.tlstm(sentence_feat)
        audio_output, _ = self.alstm(audio_feats)

        text_score = self.softmax(self.linear1(text_output))
        audio_score = self.softmax(self.linear2(audio_output))

        text_output = torch.sum(text_score * text_output, dim=1)
        audio_output = torch.sum(audio_score * audio_output, dim=1)

        # concat & norm
        if self.use_stock_data:
            all_feats = torch.cat((text_output, audio_output, stock_feats), dim=1)
        else:
            all_feats = torch.cat((text_output, audio_output), dim=1)

        feat_norm = torch.norm(all_feats, dim=1).unsqueeze(dim=-1)
        all_feats = all_feats / feat_norm

        all_feats = self.relu(self.drop(self.batchnorm(self.linear3(all_feats))))
        all_feats = self.linear4(all_feats)

        return all_feats.squeeze(-1)
