import torch
import torch.nn as nn


class CRNN(nn.Module):
    """CRNN model (see https://arxiv.org/pdf/1507.05717.pdf)."""

    def __init__(
        self,
        num_classes: int = 256,
        input_channels: int = 1,
        output_channels: int = 256,
        hidden_size: int = 256,
    ):
        """Create CRNN model."""
        super(CRNN, self).__init__()

        # Instantiate variables for feature extractor
        self.__fe = VGG(input_channels, output_channels)
        self.__fe_output = output_channels
        self.__aapool = nn.AdaptiveAvgPool2d((None, 1))

        # Instantiate variables for sequence model
        self.__sm = nn.Sequential(
            BiLSTM(self.__fe_output, hidden_size, hidden_size),
            BiLSTM(hidden_size, hidden_size, hidden_size),
        )
        self.__sm_output = hidden_size

        # Instantiate variables for CTC
        self.__predictor = nn.Linear(self.__sm_output, num_classes)

    def forward(self, x):
        """Pass input through CRNN model."""
        # Feature extraction
        viz_feat = self.__fe(x)
        viz_feat = self.__aapool(viz_feat.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        viz_feat = viz_feat.squeeze(3)

        # Sequence modelling
        ctx_feat = self.__sm(viz_feat)

        # Prediction
        return self.__predictor(ctx_feat.contiguous())


class VGG(nn.Module):
    """Feature extractor of CRNN."""

    def __init__(self, input_channels: int, output_channels: int = 512):
        """Create feature extractor."""
        super(VGG, self).__init__()

        self.__ConvNet = nn.Sequential(
            nn.Conv2d(input_channels, int(output_channels / 8), 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(int(output_channels / 8), int(output_channels / 4), 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(int(output_channels / 4), int(output_channels / 2), 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(int(output_channels / 2), int(output_channels / 2), 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(int(output_channels / 2), output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True),
            nn.Conv2d(output_channels, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(output_channels, output_channels, 2, 1, 0),
            nn.ReLU(True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through model."""
        return self.__ConvNet(x)


class BiLSTM(nn.Module):
    """Bidirectional LSTM for sequence modelling."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Create Bidirectional LSTM."""
        super(BiLSTM, self).__init__()

        # Instantiate bidirectional LSTM and linear layer
        self.__rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.__linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through model."""
        # Flatten parameters as required
        self.__rnn.flatten_parameters()

        # Pass through layers and return
        recurrent, _ = self.__rnn(x)
        x = self.__linear(recurrent)
        return x
