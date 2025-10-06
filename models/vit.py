import torch
import torch.nn as nn
import torch.nn.functional as F

class ViT(nn.Module):
    def __init__(self, num_classes, input_channels=6, input_size=256, patch_size=16, embed_dim=768, num_heads=12, depth=6):
        """
        Custom Vision Transformer for 6-channel input image classification.

        Args:
            num_classes (int): Number of output classes.
            input_channels (int): Number of input channels (default: 6).
            input_size (int): Size of the input image (must be divisible by patch_size).
            patch_size (int): Size of the patches the image is divided into.
            embed_dim (int): Embedding dimension for patch encoding.
            num_heads (int): Number of attention heads.
            depth (int): Number of Transformer layers.
        """
        super(ViT, self).__init__()

        # Ensure input size is divisible by patch size
        assert input_size % patch_size == 0, "Input size must be divisible by patch size."

        # Number of patches
        num_patches = (input_size // patch_size) ** 2

        # Patch embedding: Convolutional layer to split the image into patches
        self.patch_embed = nn.Conv2d(
            in_channels=input_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)

        # Transformer encoder: Stacked Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True  # Ensure batch is the first dimension
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # Shape: (batch_size, embed_dim, h_patches, w_patches)
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (batch_size, 1 + num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer encoding
        x = self.transformer(x)  # Shape: (batch_size, 1 + num_patches, embed_dim)

        # Classification head (use the class token output)
        cls_output = x[:, 0]  # Extract the class token's output
        out = self.mlp_head(cls_output)  # Shape: (batch_size, num_classes)

        return out