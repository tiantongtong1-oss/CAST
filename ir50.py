"""112x112 IR-50 with state-dict names matching face.evoLVe's IR_50."""

from torch import nn


class IRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        if in_channels == out_channels:
            self.shortcut_layer = nn.MaxPool2d(kernel_size=1, stride=stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class IR50(nn.Module):
    input_size = 112
    embedding_dim = 512

    def __init__(self, drop_rate=0.5):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        blocks = []
        in_channels = 64
        for out_channels, depth in ((64, 3), (128, 4), (256, 14), (512, 3)):
            blocks.append(IRBlock(in_channels, out_channels, stride=2))
            blocks.extend(IRBlock(out_channels, out_channels, stride=1)
                          for _ in range(depth - 1))
            in_channels = out_channels
        self.body = nn.Sequential(*blocks)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(drop_rate),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
        )
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.output_layer(self.body(self.input_layer(x)))
