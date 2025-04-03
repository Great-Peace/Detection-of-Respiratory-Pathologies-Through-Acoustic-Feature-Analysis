import torch
import torch.nn as nn


class CNNLSTMClassifier(nn.Module):
    """
    CNN-LSTM model for classifying MFCC-based audio inputs.

    Input shape: (batch_size, 1, n_mfcc, time_steps)
    Output: Binary class probability (COVID vs. Non-COVID)
    """

    def __init__(self, n_mfcc: int = 40, time_steps: int = 157, lstm_hidden: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (B, 16, 40, T)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),            # (B, 16, 20, T//2)
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # (B, 32, 20, T//2)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),            # (B, 32, 10, T//4)
        )

        # Infer output shape for LSTM
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_mfcc, time_steps)
            cnn_out = self.cnn(dummy_input)  # shape: (1, 32, 10, T')
            b, c, h, w = cnn_out.shape
            self.lstm_input_size = h * c  # (32 * 10)
            self.lstm_time_steps = w      # (T//4)

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, 1, n_mfcc, time_steps)
        Returns:
            Tensor: (B, 1) probabilities
        """
        cnn_out = self.cnn(x)                   # (B, 32, 10, T//4)
        b, c, h, w = cnn_out.shape
        rnn_in = cnn_out.permute(0, 3, 1, 2)    # (B, T', C, H)
        rnn_in = rnn_in.reshape(b, w, c * h)    # (B, T', C*H)
        lstm_out, _ = self.lstm(rnn_in)         # (B, T', 2 * hidden)
        last_hidden = lstm_out[:, -1, :]        # (B, 2 * hidden)
        return self.classifier(last_hidden)
