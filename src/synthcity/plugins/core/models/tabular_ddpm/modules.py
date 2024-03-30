"""
Code was adapted from https://github.com/Yura52/rtdl
"""
# stdlib
import math
from typing import Optional, Union

# third party
import torch
import torch.optim
from torch import Tensor, nn

# synthcity absolute
from synthcity.plugins.core.models.factory import get_model, get_nonlin


class TimeStepEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_period: int = 10000,
        n_layers: int = 2,
        nonlin: Union[str, nn.Module] = "silu",
    ) -> None:
        """
        Create sinusoidal timestep embeddings.

        Args:
        - dim (int): the dimension of the output.
        - max_period (int): controls the minimum frequency of the embeddings.
        - n_layers (int): number of dense layers
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.n_layers = n_layers

        if dim % 2 != 0:
            raise ValueError(f"embedding dim must be even, got {dim}")

        layers = []
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(dim, dim))
            layers.append(get_nonlin(nonlin))

        self.fc = nn.Sequential(*layers, nn.Linear(dim, dim))

    def forward(self, timesteps: Tensor) -> Tensor:
        """
        Args:
        - timesteps (Tensor): 1D Tensor of N indices, one per batch element.
        """
        d, T = self.dim, self.max_period
        mid = d // 2
        fs = torch.exp(-math.log(T) / mid * torch.arange(mid, dtype=torch.float32))
        fs = fs.to(timesteps.device)
        args = timesteps[:, None].float() * fs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.fc(emb)

# class SinusoidalEmbedding(nn.Module):
    
#     def __init__(self, embedding_dim=128):
#         super(SinusoidalEmbedding, self).__init__()
#         self.embedding_dim = embedding_dim
        
#     def _generate_sinusoidal_embeddings(self, time_to_event):
#         batch_size = time_to_event.size(0)
#         embeddings = torch.zeros(batch_size, self.embedding_dim)

#         # Assign sinusoidal embeddings based on the order of time-to-event
#         for j in range(self.embedding_dim // 2):
#             embeddings[:, 2 * j] = torch.sin(time_to_event.squeeze(-1) / (10000 ** (2 * j / self.embedding_dim)))
#             embeddings[:, 2 * j + 1] = torch.cos(time_to_event.squeeze(-1) / (10000 ** (2 * j / self.embedding_dim)))

#         return embeddings

#     def forward(self, input_data):
#         batch_size = input_data.size(0)
#         time_to_event = input_data[:, 0].unsqueeze(1)  # Extract time-to-event
#         event_indicators = input_data[:, 1].unsqueeze(1)  # Extract event indicators

#         # Create sinusoidal embeddings for time-to-event
#         time_embeddings = self._generate_sinusoidal_embeddings(time_to_event)

#         # Concatenate event indicators to sinusoidal embeddings
#         # value_embeddings = torch.cat((time_embeddings, event_indicators), dim=1)

#         return time_embeddings

# class SinusoidalEmbedding(nn.Module):
    
#     def __init__(self, embedding_dim=128):
#         super(SinusoidalEmbedding, self).__init__()
#         self.embedding_dim = embedding_dim
        
#     def _generate_sinusoidal_embeddings(self, time_to_event):
#         batch_size = time_to_event.size(0)
#         embeddings = torch.zeros(batch_size, self.embedding_dim)

#         # Assign sinusoidal embeddings based on the order of time-to-event
#         for j in range(self.embedding_dim // 2):
#             embeddings[:, 2 * j] = torch.sin(time_to_event / (10000 ** (2 * j / self.embedding_dim)))
#             embeddings[:, 2 * j + 1] = torch.cos(time_to_event / (10000 ** (2 * j / self.embedding_dim)))

#         return embeddings

#     def forward(self, input_data):
#         batch_size = input_data.size(0)
#         time_to_event = input_data[:, 1]  # Extract time-to-event
#         event_indicators = input_data[:, 0].unsqueeze(1)  # Extract event indicators

#         # Create sinusoidal embeddings for time-to-event
#         time_embeddings = self._generate_sinusoidal_embeddings(time_to_event)

#         # Concatenate event indicators to sinusoidal embeddings
#         value_embeddings = torch.cat((event_indicators, time_embeddings), dim=1)

#         return value_embeddings

class SinusoidalAndEmbeddingLayer(nn.Module):
    def __init__(self, dim_emb: int, max_time_period: int, num_classes: int):
        super().__init__()
        self.dim_emb = dim_emb
        self.max_time_period = max_time_period
        self.num_classes = num_classes

        self.event_emb = nn.Embedding(num_classes, dim_emb // 2)

    def forward(self, inputs: Tensor) -> Tensor:
        time_to_event, event_indicator = inputs[:, 0], inputs[:, 1]

        # Sort time-to-event values
        sorted_indices = torch.argsort(time_to_event)
        time_to_event = time_to_event[sorted_indices]

        # Calculate sinusoidal embeddings
        dim = self.dim_emb // 2
        max_period = self.max_time_period
        freqs = torch.exp(-math.log(max_period) / (dim - 1) * torch.arange(dim, dtype=torch.float32, device=time_to_event.device))
        args = time_to_event[:, None] * freqs[None, :]
        sinusoidal_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # Unsort sinusoidal embeddings
        sinusoidal_emb = sinusoidal_emb[sorted_indices.argsort()]

        event_emb = self.event_emb(event_indicator.long())

        return torch.cat([sinusoidal_emb, event_emb], dim=-1)

# class DiffusionModel(nn.Module):
#     def __init__(
#         self,
#         dim_in: int,
#         dim_emb: int = 128,
#         *,
#         model_type: str = "mlp",
#         model_params: dict = {},
#         conditional: bool = False,
#         num_classes: int = 0,
#         emb_nonlin: Union[str, nn.Module] = "silu",
#         max_time_period: int = 10000,
#     ) -> None:
#         super().__init__()
#         self.dim_t = dim_emb
#         self.num_classes = num_classes
#         self.has_label = conditional

#         if isinstance(emb_nonlin, str):
#             self.emb_nonlin = get_nonlin(emb_nonlin)
#         else:
#             self.emb_nonlin = emb_nonlin

#         self.proj = nn.Linear(dim_in, dim_emb)
#         self.time_emb = TimeStepEmbedding(dim_emb, max_time_period)

#         if conditional:
#             if self.num_classes > 0:
#                 # self.label_emb = nn.Embedding(self.num_classes, dim_emb)
#                 self.label_emb= SinusoidalEmbedding(embedding_dim=128)
#             elif self.num_classes == 0:  # regression
#                 self.label_emb = nn.Linear(1, dim_emb)

#         if not model_params:
#             model_params = {}  # avoid changing the default dict

#         if model_type == "mlp":
#             if not model_params:
#                 model_params = dict(n_units_hidden=256, n_layers_hidden=3, dropout=0.0)
#             model_params.update(n_units_in=dim_emb, n_units_out=dim_in)
#         elif model_type == "tabnet":
#             model_params.update(input_dim=dim_emb, output_dim=dim_in)

#         self.model = get_model(model_type, model_params)

#     def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
#         emb = self.time_emb(t)
#         print("Size of x:", x.size())
#         print("Size of t:", t.size())
#         if y is not None:
#             print("Size of y:", y.size())
           
#         if self.has_label:
#             try:
#                 z=self.emb_nonlin(self.label_emb(y))
#                 print("Size of z:", z.size())
#             except:
                
#                 if y is None:
#                     raise ValueError("y must be provided if conditional is True")
#                 if self.num_classes == 0:
#                     y = y.reshape(-1, 1).float()
#                 # else:
#                     # y = y.squeeze().long()
#                     # y=y.long()
#                 # print(y.size())
#                 # print("Size of y here is :", y.size())
                
                
#                 # emb += self.emb_nonlin(self.label_emb(y))
#                 # print(emb.size())
#                 emb += z
#         x = self.proj(x) + emb
#         return self.model(x)

# class DiffusionModel(nn.Module):
#     def __init__(
#         self,
#         dim_in: int,
#         dim_emb: int = 128,
#         *,
#         model_type: str = "mlp",
#         model_params: dict = {},
#         conditional: bool = False,
#         num_classes: int = 0,
#         emb_nonlin: Union[str, nn.Module] = "silu",
#         max_time_period: int = 10000,
#     ) -> None:
#         super().__init__()
#         self.dim_t = dim_emb
#         self.num_classes = num_classes
#         self.has_label = conditional

#         if isinstance(emb_nonlin, str):
#             self.emb_nonlin = get_nonlin(emb_nonlin)
#         else:
#             self.emb_nonlin = emb_nonlin

#         self.proj = nn.Linear(dim_in, dim_emb)
#         self.time_emb = TimeStepEmbedding(dim_emb, max_time_period)

#         if conditional:
#             if self.num_classes > 0:
#                 self.label_emb = SinusoidalEmbedding(embedding_dim=dim_emb + 1)
#             elif self.num_classes == 0:  # regression
#                 self.label_emb = nn.Linear(2, dim_emb)

#         if not model_params:
#             model_params = {}  # avoid changing the default dict

#         if model_type == "mlp":
#             if not model_params:
#                 model_params = dict(n_units_hidden=256, n_layers_hidden=3, dropout=0.0)
#             model_params.update(n_units_in=dim_emb, n_units_out=dim_in)
#         elif model_type == "tabnet":
#             model_params.update(input_dim=dim_emb, output_dim=dim_in)

#         self.model = get_model(model_type, model_params)

#     def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
#         emb = self.time_emb(t)
#         print("Size of x:", x.size())
#         print("Size of t:", t.size())
#         if y is not None:
#             print("Size of y:", y.size())
#             print("First 5 examples of y:")
#             print(y[:5])

#         if self.has_label:
#             if y is None:
#                 raise ValueError("y must be provided if conditional is True")
#             z = self.emb_nonlin(self.label_emb(y))
#             print(y[:,0:5])
#             print("Size of z:", z.size())
#             print(z[:5])
#             # emb = torch.cat((emb, z), dim=-1)
#             emb += self.emb_nonlin(self.label_emb(y))

#         x = self.proj(x) + emb
#         return self.model(x)

class DiffusionModel(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_emb: int = 128,
        *,
        model_type: str = "mlp",
        model_params: dict = {},
        conditional: bool = False,
        num_classes: int = 0,
        emb_nonlin: Union[str, nn.Module] = "silu",
        max_time_period: int = 10000,
    ) -> None:
        super().__init__()
        self.dim_t = dim_emb
        self.num_classes = num_classes
        self.has_label = conditional

        if isinstance(emb_nonlin, str):
            self.emb_nonlin = get_nonlin(emb_nonlin)
        else:
            self.emb_nonlin = emb_nonlin

        self.proj = nn.Linear(dim_in, dim_emb)
        self.time_emb = TimeStepEmbedding(dim_emb, max_time_period)

        if conditional:
            self.label_emb = SinusoidalAndEmbeddingLayer(dim_emb, max_time_period, num_classes)

        if not model_params:
            model_params = {}  # avoid changing the default dict

        if model_type == "mlp":
            if not model_params:
                model_params = dict(n_units_hidden=256, n_layers_hidden=3, dropout=0.0)
            model_params.update(n_units_in=dim_emb, n_units_out=dim_in)
        elif model_type == "tabnet":
            model_params.update(input_dim=dim_emb, output_dim=dim_in)

        self.model = get_model(model_type, model_params)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.time_emb(t)
        if self.has_label:
            if y is None:
                raise ValueError("y must be provided if conditional is True")
            emb += self.emb_nonlin(self.label_emb(y))
        x = self.proj(x) + emb
        return self.model(x)
