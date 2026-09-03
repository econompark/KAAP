import math

from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class TransformerConfig:
    """
    Vanilla Transformer
    """

    input_dim:          int
    target_dim:         int = 1

    d_model:            int = 128
    num_heads:          int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    d_ff:               int = 512

    max_seq_len:        int = 512
    dropout:            float = 0.1

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be greater than 0.")

        if self.target_dim <= 0:
            raise ValueError("target_dim must be greater than 0.")

        if self.d_model <= 0:
            raise ValueError("d_model must be greater than 0.")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be greater than 0.")

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")


if __name__ == "__main__":
    config = TransformerConfig(
        input_dim  = 10,
        target_dim = 1,
        d_model    = 128,
        num_heads  = 8,
    )

    print(config)

class InputEmbedding(nn.Module):
    """
    Projects continuous input features into the model dimension.
    """

    def __init__(self, input_dim: int, d_model: int,) -> None:
        super().__init__()
        self.d_model    = d_model
        self.projection = nn.Linear(in_features = input_dim, out_features = d_model,)

    def forward(self, x: torch.Tensor,) -> torch.Tensor:
        x = self.projection(x)
        x = x * math.sqrt(self.d_model)

        return x

class PositionalEncoding(nn.Module):
    """
    Add sinusoidal positional information to input embeddings
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float,) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_seq_len, dtype = torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype = torch.float32,) * (-math.log(10000.0) / d_model))
        
        positional_encoding = torch.zeros(max_seq_len, d_model)

        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term[: positional_encoding[:, 1::2].shape[1]])

        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.size(1)

        if sequence_length > self.positional_encoding.size(1):
            raise ValueError("Input sequence length exceeds max_seq_len.")

        x = (x + self.positional_encoding[:, :sequence_length, :])

        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    """
    Computes scaled dot-product attention.
    """

    def __init__(self, dropout: float = 0.0,) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:

        d_k = query.size(-1)

        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1)
        )

        attention_scores = (attention_scores / math.sqrt(d_k))

        if mask is not None:
            mask = mask.to(dtype = torch.bool)

            attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim = -1)

        dropped_attention_weights = self.dropout(attention_weights)

        output = torch.matmul(dropped_attention_weights, value,)

        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Compute attention over multiple representation heads.
    """

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads

        self.query_projection = nn.Linear(
            d_model,
            d_model
        )

        self.key_projection = nn.Linear(
            d_model,
            d_model
        )

        self.value_projection = nn.Linear(
            d_model,
            d_model
        )

        self.attention = ScaledDotProductAttention(
            dropout = dropout
        )

        self.output_projection = nn.Linear(
            d_model,
            d_model
        )

    def _split_heads(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        sequence_length = x.size(1)

        x = x.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        x = x.transpose(1, 2)

        return x

    def _combine_heads(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        sequence_length = x.size(2)

        x = x.transpose(1, 2).contiguous()

        x = x.reshape(
            batch_size,
            sequence_length,
            self.d_model
        )

        return x

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    
        projected_query = self.query_projection(query)
        projected_key   = self.key_projection(key)
        projected_value = self.value_projection(value)

        split_query = self._split_heads(projected_query)
        split_key   = self._split_heads(projected_key)
        split_value = self._split_heads(projected_value)

        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        attention_output, attention_weights = (
            self.attention(
                query = split_query,
                key   = split_key,
                value = split_value,
                mask  = mask
            )
        )

        combined_output = self._combine_heads(
            attention_output
        )

        output = self.output_projection(
            combined_output
        )

        return output, attention_weights

class PositionWiseFeedForward(nn.Module):
    """
    Applies the same feed-forward network
    independently to every sequence position.
    """

    def __init__(
            self,
            d_model: int,
            d_ff:    int,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(
            d_model,
            d_ff,
        )

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.output_projection = nn.Linear(
            d_ff,
            d_model,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        output = self.output_projection(hidden)

        return output

class AddAndNorm(nn.Module):
    """
    Applied dropout, residual addition, and layer normalization
    """

    def __init__(self, d_model: int, dropout: float = 0.0,) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        if x.shape != sublayer_output.shape:
            raise ValueError(
                "x and sublayer_output must have the same shape"
            )

        residual = x + self.dropout(sublayer_output)

        output   = self.layer_norm(residual)

        return output

class EncoderLayer(nn.Module):
    """
    A single Vanilla Transformer encoder layer.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model = d_model,
            num_heads = num_heads,
            dropout = dropout,
        )

        self.attention_add_norm = AddAndNorm(
            d_model = d_model,
            dropout = dropout,
            )

        self.feed_forward = PositionWiseFeedForward(
            d_model = d_model,
            d_ff    = d_ff,
            dropout = dropout
        )

        self.feed_forward_add_norm = AddAndNorm(
            d_model = d_model,
            dropout = dropout,
        )

    def forward(
            self, x: torch.Tensor, mask: torch.Tensor | None = None
            ) -> tuple[torch.Tensor, torch.Tensor]:
        
        attention_output, attention_weights = (self.self_attention(query = x, key = x, value = x, mask = mask))
        x = self.attention_add_norm(x = x, sublayer_output = attention_output)

        feed_forward_output = self.feed_forward(x)
        x = self.feed_forward_add_norm(x = x, sublayer_output = feed_forward_output)

        return x, attention_weights

class TransformerEncoder(nn.Module):
    """
    Stacks multiple Transformer encoder layers.
    """

    def __init__(
        self,
        d_model:    int,
        num_heads:  int,
        d_ff:       int,
        num_layers: int,
        dropout:    float = 0.0,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model   = d_model,
                num_heads = num_heads,
                d_ff      = d_ff,
                dropout   = dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:

        attention_weights_list = []

        for layer in self.layers:
            x, attention_weights = layer(x = x, mask = mask,)

            attention_weights_list.append(attention_weights)

        return x, attention_weights_list

class DecoderLayer(nn.Module):
    """
    A single Vanilla Transformer decoder layer.
    """

    def __init__(
            self,
            d_model:  int,
            num_heads: int,
            d_ff:      int,
            dropout:   float = 0.0,
    ) -> None:
        super().__init__()

        # Masked self-attention
        self.self_attention = MultiHeadAttention(
            d_model   = d_model,
            num_heads = num_heads,
            dropout   = dropout,
        )

        self.self_attention_add_norm = AddAndNorm(
            d_model = d_model,
            dropout = dropout,
        )

        # Cross-attention
        self.cross_attention = MultiHeadAttention(
            d_model = d_model,
            num_heads = num_heads,
            dropout = dropout,
        )

        self.cross_attention_add_norm = AddAndNorm(
            d_model = d_model,
            dropout = dropout,
        )

        # Position-wise feed-forward
        self.feed_forward = PositionWiseFeedForward(
            d_model = d_model,
            d_ff    = d_ff,
            dropout = dropout
        )

        self.feed_forward_add_norm = AddAndNorm(
            d_model = d_model,
            dropout = dropout
        )

    def forward(
            self,
            x: torch.Tensor,
            encoder_output: torch.Tensor,
            target_mask: torch.Tensor | None = None,
            source_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        self_attention_output, self_attention_weights = (
            self.self_attention(
                query = x,
                key   = x,
                value = x,
                mask  = target_mask
            )
        )

        x = self.self_attention_add_norm(
            x = x,
            sublayer_output = self_attention_output,
        )

        cross_attention_output, cross_attention_weights = (
            self.cross_attention(
                query = x,
                key   = encoder_output,
                value = encoder_output,
                mask  = source_mask,
            )
        )

        x = self.cross_attention_add_norm(
            x = x,
            sublayer_output = cross_attention_output,
        )

        feed_forward_output = self.feed_forward(x)

        x = self.feed_forward_add_norm(
            x = x,
            sublayer_output = feed_forward_output,
        )

        return x, self_attention_weights, cross_attention_weights

class TransformerDecoder(nn.Module):
    """
    Stacks multiple Transformer decoder layers.
    """

    def __init__(
            self,
            d_model:    int,
            num_heads:  int,
            d_ff:       int,
            num_layers: int,
            dropout:    float = 0.0,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model   = d_model,
                num_heads = num_heads,
                d_ff      = d_ff,
                dropout   = dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
            self,
            x: torch.Tensor,
            encoder_output: torch.Tensor,
            target_mask: torch.Tensor | None = None,
            source_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        self_attention_weights_list  = []
        cross_attention_weights_list = []

        # note that the decoder layers use the same encoder output.
        for layer in self.layers:
            x, self_attention_weights, cross_attention_weights = layer(
                x = x,
                encoder_output = encoder_output,
                target_mask = target_mask,
                source_mask = source_mask,
            )

            self_attention_weights_list.append(
                self_attention_weights
            )
            cross_attention_weights_list.append(
                cross_attention_weights
            )

        return (x, self_attention_weights_list, cross_attention_weights_list)

class VanillaTransformer(nn.Module):
    """
    Complete encoder-decoder Vanilla Transformer.
    Input for the encoder: source
    Input for the decoder: encoder output + shifted target
    """

    def __init__(
            self,
            config: TransformerConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.source_embedding = InputEmbedding(
            input_dim = config.input_dim,
            d_model   = config.d_model,
        )

        self.target_embedding = InputEmbedding(
            input_dim = config.target_dim,
            d_model   = config.d_model,
        )

        self.source_positional_encoding = PositionalEncoding(
            d_model      = config.d_model,
            max_seq_len  = config.max_seq_len,
            dropout      = config.dropout,
        )

        self.target_positional_encoding = PositionalEncoding(
            d_model     = config.d_model,
            max_seq_len = config.max_seq_len,
            dropout     = config.dropout
        )

        self.encoder = TransformerEncoder(
            d_model    = config.d_model,
            num_heads  = config.num_heads,
            d_ff       = config.d_ff,
            num_layers = config.num_encoder_layers,
            dropout    = config.dropout,
        )

        self.decoder = TransformerDecoder(
            d_model    = config.d_model,
            num_heads  = config.num_heads,
            d_ff       = config.d_ff,
            num_layers = config.num_decoder_layers,
            dropout    = config.dropout,
        )

        self.output_projection = nn.Linear(
            in_features  = config.d_model,
            out_features = config.target_dim,
        )

    def encode(
            self,
            source: torch.Tensor,
            source_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        
        source = self.source_embedding(source)

        source = self.source_positional_encoding(source)

        encoder_output, encoder_attention_weights = (
            self.encoder(
                x = source,
                mask = source_mask,
            )
        )

        return encoder_output, encoder_attention_weights

    def decode(
            self,
            decoder_input: torch.Tensor,
            encoder_output: torch.Tensor,
            target_mask: torch.Tensor | None = None,
            source_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor]
    ]:
        decoder_input = self.target_embedding(
            decoder_input

        )

        decoder_input = self.target_positional_encoding(
            decoder_input
        )

        (
            decoder_output,
            self_attention_weights,
            cross_attention_weights,
        ) = self.decoder(
            x              = decoder_input,
            encoder_output = encoder_output,
            target_mask    = target_mask,
            source_mask    = source_mask,
        )

        return (
            decoder_output,
            self_attention_weights,
            cross_attention_weights,
        )

    def forward(
            self,
            source: torch.Tensor,
            decoder_input: torch.Tensor,
            source_mask: torch.Tensor | None = None,
            target_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor]
    ]:
        
        if target_mask is None:
            target_mask = self.generate_causal_mask(
                    sequence_length = decoder_input.size(1),
                    device = decoder_input.device
                )
            
        encoder_output, encoder_attention_weights = (
            self.encode(
                source = source,
                source_mask = source_mask,
            )
        )

        (
            decoder_output,
            decoder_self_attention_weights,
            decoder_cross_attention_weights,
        ) = self.decode(
            decoder_input  = decoder_input,
            encoder_output = encoder_output,
            target_mask    = target_mask,
            source_mask    = source_mask,
        )

        prediction = self.output_projection(
            decoder_output
        )

        return (
            prediction,
            encoder_attention_weights,
            decoder_self_attention_weights,
            decoder_cross_attention_weights,
        )

    @staticmethod
    def generate_causal_mask(
        sequence_length: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be greater than 0."
            )

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype = torch.bool,
                device = device,
            )
        )

        return mask