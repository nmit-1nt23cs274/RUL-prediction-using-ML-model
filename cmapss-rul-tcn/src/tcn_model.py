from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.residual = nn.Identity()
        if n_inputs != n_outputs:
            self.residual = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        return F.relu(out + residual)


class TCN(nn.Module):
    def __init__(self, input_size: int, num_channels, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.num_channels = list(num_channels)
        self.kernel_size = kernel_size
        self.dropout = dropout

        layers = []
        for i, out_channels in enumerate(self.num_channels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else self.num_channels[i - 1]
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(self.num_channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    @property
    def receptive_field(self) -> int:
        field = 1
        for i in range(len(self.num_channels)):
            field += (self.kernel_size - 1) * (2 ** i) * 2
        return field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x[:, :, -1]
        out = self.head(x)
        return out.squeeze(-1)
