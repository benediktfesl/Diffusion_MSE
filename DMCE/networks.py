import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
import math
import warnings
from typing import Tuple, Union

from DMCE import utils


def get_positional_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Creates the DM time embedding from an integer time step

    Parameters
    ----------
    t : Tensor of shape [batch_size]
        timesteps of the corresponding data samples
    dim : int
        dimension of the resulting embedding
    Returns
    -------
    t_emb : Tensor of shape [batch_size, dim]
        time embeddings for each data sample
    """

    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(- emb * torch.arange(half_dim, device=t.device))
    emb = t[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

    # if dim is an odd number, pad the last entry of the embedding vector with zeros
    if dim % 2 != 0:
        emb = torch.nn.functional.pad(emb, (0, 1), 'constant', 0)
    return emb


def get_normalization_layer(norm_type: str, num_channels: int, num_groups: int = None,
                            device: Union[str, torch.device] = 'cuda'):
    """
    provides access to the normalization layer specified by 'norm_type'
    """
    if norm_type == 'group':
        return nn.GroupNorm(num_channels=num_channels, num_groups=num_groups, device=device)
    elif norm_type == 'batch':
        return BatchNormND(num_features=num_channels, device=device)
    elif norm_type == 'instance':
        return InstanceNormND(num_features=num_channels, device=device)
    else:
        raise NotImplementedError(norm_type)


class BatchNormND(_BatchNorm):
    """
    child class of PyTorch '_BatchNorm' class, that overrides the '_check_input_dim()' function
    """
    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError(
                "expected 2D or bigger input (got {}D input)".format(input.dim())
            )


class InstanceNormND(_InstanceNorm):
    """
    child class of PyTorch '_BatchNorm' class, that overrides the '_check_input_dim()' and '_bet_no_batch_dim' functions
    """
    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError(
                "expected 2D or bigger input (got {}D input)".format(input.dim())
            )

    def _get_no_batch_dim(self):
        return 0


class Upsample(nn.Module):
    def __init__(self,
                 ch_in: int,
                 *,
                 ch_out: int = None,
                 scaling_factor: int = 2,
                 mode: str = '1D',
                 with_conv: bool = False,
                 kernel_size: int = 3,
                 device: Union[str, torch.device] = 'cuda'):
        """
        Upsampling layer for the U-Net architecture. The 'forward' method multiplies the feature dimensions of a data
        sample by a scaling factor, usually 2.

        Parameters
        ----------
        ch_in : int
            Number of input channels
        ch_out : optional int
            Number of requested output channels. Can only be different if 'with_conv' is True.  Defaults to 'ch_in'
            if not given.
        scaling_factor : int
            factor by which the feature dimensions are scaled
        mode : str {'1D', '2D'}
            Identifies number of feature dimensions, disregarding the channel dimension. (should rather be implemented
            as an int)
        with_conv : bool
            Specifies, whether the Upsampling step should also include a convolutional layer or not
        kernel_size : int
            kernel size of the convolutional layer, if it is included
        device : torch.device or str {'cpu', 'cuda', 'cuda:id', ...}
            Device on which the NN is working.
        """

        super().__init__()
        self.device = utils.set_device(device)
        self.scaling_factor = scaling_factor
        self.ch_in = ch_in

        ch_out = utils.default(ch_out, ch_in)
        if not with_conv:
            assert ch_out == ch_in
        self.ch_out = ch_out

        if mode == '1D':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scaling_factor, mode='nearest'),
                nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, padding='same', device=self.device) if with_conv else nn.Identity()
            )
        elif mode == '2D':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scaling_factor, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=(kernel_size, kernel_size), padding='same', device=self.device) if with_conv else nn.Identity()
            )
        else:
            raise ValueError(f'Upsample mode {mode} is not supported.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Upsampling layer.

        Parameters
        ----------
        x : Tensor of shape [batch_size, ch_in, dim1, dim2, ...]
            batched Input Tensor

        Returns
        -------
        x : Tensor of shape [batch_size, ch_out, 2 * dim1, 2 * dim2, ...]
            batched Output Tensor
        """

        bs, c, *data_sizes = x.shape

        assert c == self.ch_in

        data_sizes_expected = [self.scaling_factor * size for size in data_sizes]
        x = self.upsample(x)
        assert x.shape == torch.Size((bs, self.ch_out, *data_sizes_expected))
        return x


class Downsample(nn.Module):
    def __init__(self,
                 ch_in: int,
                 ch_out: int = None,
                 dividing_factor: int = 2,
                 mode: str = '1D',
                 with_conv: bool = False,
                 device: Union[str, torch.device] = 'cuda'):
        """
        Downsampling layer for the U-Net architecture. The 'forward()' method divides the feature dimensions of an Input
        Tensor by the 'dividing_factor', usually 2. The downsampling is performed by either a convolutional layer or
        an average pooling layer.

        Parameters
        ----------
        ch_in : int
            Number of input channels
        ch_out : optional int
            Number of requested output channels. Can only be different if 'with_conv' is True.  Defaults to 'ch_in'
            if not given.
        dividing_factor : int
            Factor by which each feature dimension is divided. Directly converts to the 'stride' of the convolutional
            or average pooling layer
        mode : str {'1D', '2D'}
            Identifies number of feature dimensions, disregarding the channel dimension. (should rather be implemented
            as an int)
        with_conv : bool
            Specifies, whether the downsampling operation should be performed by a convolutional layer by choosing an
            appropriate stride ar an Average Pooling layer.
        device : torch.device or str {'cpu', 'cuda', 'cuda:id', ...}
            Device on which the NN is working.
        """

        super().__init__()
        self.device = utils.set_device(device)
        self.dividing_factor = dividing_factor
        self.ch_in = ch_in

        ch_out = utils.default(ch_out, ch_in)
        self.ch_out = ch_out

        if not with_conv:
            assert ch_out == ch_in

        if mode == '1D':
            if with_conv:
                self.downsample = nn.Conv1d(ch_in, ch_out, kernel_size=dividing_factor, stride=dividing_factor,
                                            device=self.device)
            else:
                self.downsample = nn.AvgPool1d(kernel_size=dividing_factor, stride=dividing_factor)
        elif mode == '2D':
            if with_conv:
                self.downsample = nn.Conv2d(ch_in, ch_out, kernel_size=dividing_factor, stride=dividing_factor,
                                            device=self.device)
            else:
                self.downsample = nn.AvgPool2d(kernel_size=dividing_factor, stride=dividing_factor)
        else:
            raise ValueError(f'Upsample mode {mode} is not supported.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Downsampling layer.

        Parameters
        ----------
        x : Tensor of shape [batch_size, ch_in, dim1, dim2, ...]
            batched Input Tensor

        Returns
        -------
        x : Tensor of shape [batch_size, ch_out, dim1 // self.dividing_factor, dim2 // self.dividing_factor, ...]
            batched Output Tensor
        """

        bs, c, *data_sizes = x.shape

        assert c == self.ch_in

        for dim, size in enumerate(data_sizes):
            if size % self.dividing_factor != 0:
                warnings.warn(f'Downsampling Problem: Feature dimensions {data_sizes} are not divisible by dividing'
                              f' factor {self.dividing_factor}.')
        data_size_expected = [size // self.dividing_factor for size in data_sizes]

        x = self.downsample(x)
        assert x.shape == torch.Size((bs, self.ch_out, *data_size_expected))
        return x


class ResNetBlock(nn.Module):
    def __init__(self, ch_in: int,
                 ch_out: int = None,
                 *,
                 time_emb_dim: int,
                 kernel_size: int = 3,
                 mode: str = '1D',
                 dropout: float = 0.,
                 norm_type: str = 'batch',
                 num_groups: int = 8,
                 device: Union[str, torch.device] = 'cuda'):
        """

        Parameters
        ----------
        ch_in : int
            number of input channels of the ResNet block
        ch_out : int
            number of output channels of the ResNet block. Defaults to 'ch_in'
        time_emb_dim : int
            dimension of the time embedding
        kernel_size : int
            kernel size of the convolutional layers in the ResNet block
        mode : str {'1D', '2D'}
            Identifies number of feature dimensions, disregarding the channel dimension. (should rather be implemented
            as an int)
        dropout : float
            percentage of dropped samples in Dropout layer during training
        norm_type : str {'batch', 'group', 'instance'}
            type of normalization layer that is used
        num_groups : int
            if the group normalization is used, this specifies the number of normalization groups
        device : torch.device or str {'cpu', 'cuda', 'cuda:id', ...}
            Device on which the NN is working.
        """

        super().__init__()

        self.device = utils.set_device(device)
        self.mode = mode

        self.ch_out = utils.default(ch_out, ch_in)

        # First normalization layer
        if norm_type == 'group':
            assert ch_in % num_groups == 0 and self.ch_out % num_groups == 0, \
                f'number of input channels must be divisible by number of normalization groups'
        norm1 = get_normalization_layer(norm_type=norm_type, num_channels=ch_in,
                                        num_groups=num_groups, device=self.device)

        # First SiLU layer
        silu1 = nn.SiLU()

        # First convolutional layer, converting number of channels from 'ch_in' to 'ch_out'
        if mode == '1D':
            conv1 = nn.Conv1d(ch_in, self.ch_out, kernel_size=kernel_size, padding='same', device=self.device)
        elif mode == '2D':
            conv1 = nn.Conv2d(ch_in, self.ch_out, kernel_size=kernel_size, padding='same', device=self.device)
        else:
            raise ValueError(f'Upsample mode {mode} is not supported.')

        self.block1 = nn.Sequential(norm1, silu1, conv1)

        # Time Embedding is converted to correct dimension with SiLU and linear layer
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * self.ch_out, device=self.device)
        )

        # second normalization layer
        norm2 = get_normalization_layer(norm_type=norm_type, num_channels=self.ch_out,
                                        num_groups=num_groups, device=self.device)

        # second SiLU layer
        silu2 = nn.SiLU()

        # second convolutional layer, possibly followed by a Dropout layer
        if mode == '1D':
            conv2 = nn.Conv1d(self.ch_out, self.ch_out, kernel_size=kernel_size, padding='same', device=self.device)
            dropout = nn.Dropout1d(p=dropout)
        elif mode == '2D':
            conv2 = nn.Conv2d(self.ch_out, self.ch_out, kernel_size=kernel_size, padding='same', device=self.device)
            dropout = nn.Dropout2d(p=dropout)
        else:
            raise ValueError(f'Upsample mode {mode} is not supported.')

        self.block2 = nn.Sequential(norm2, silu2, dropout, conv2)

        # bypass adds the input to the result after the second block, which then yields the result
        if ch_in != self.ch_out:
            if mode == '1D':
                self.bypass = nn.Conv1d(ch_in, self.ch_out, kernel_size, padding='same', device=self.device)
            elif mode == '2D':
                self.bypass = nn.Conv2d(ch_in, self.ch_out, kernel_size, padding='same', device=self.device)
            else:
                raise ValueError(f'Upsample mode {mode} is not supported.')

        else:
            self.bypass = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ResNet block.

        Parameters
        ----------
        x : Tensor of shape [batch_size, ch_in, dim1, dim2, ...]
            batched Input tensor for the ResNet block
        t_emb : Tensor of shape [batch_size, time_emb_dim]
            base time embedding vectors

        Returns
        -------
        x : Tensor of shape [batch_size, ch_out, dim1, dim2, ...]
            batched Output Tensor with possibly a different number of channels than the Input Tensor
        """

        # perform time embedding calculation and extract scaling and shifting vectors
        t_emb = self.time_mlp(t_emb)
        scale = t_emb[:, :self.ch_out]
        shift = t_emb[:, self.ch_out:]

        h = self.block1(x)
        assert scale.shape == h.shape[:2] and shift.shape == h.shape[:2]

        # scale and shifting the features depending on the time embedding introduces the time information
        if self.mode == '1D':
            h = h + scale[:, :, None] * h + shift[:, :, None]
        elif self.mode == '2D':
            h = h + scale[:, :, None, None] * h + shift[:, :, None, None]
        else:
            raise ValueError(f'Upsample mode {self.mode} is not supported.')

        h = self.block2(h)
        return h + self.bypass(x)


class UNet(nn.Module):
    def __init__(self,
                 ch_data: int,
                 ch_init: int = 16,
                 ch_out: int = None,
                 kernel_size: int = 3,
                 mode: str = '1D',
                 ch_multipliers: Tuple = (1, 2, 4),
                 num_res_blocks: int = 2,
                 dropout: float = 0.,
                 norm_type: str = 'batch',
                 device: Union[str, torch.device] = 'cuda'):
        """
        Full U-Net architecture also implementing residual connections typical for a ResNet architecture. As it stands,
        the implementation can be applied to 1 and 2-dimensional data. By specifying different parameters, the
        network can be flexibly re-structured.

        Parameters
        ----------
        ch_data : int
            Number of channels that the input data has
        ch_init : int
            Number of channels after the initial convolutional layer
        ch_out : int
            Number of requested output channels. Defaults to 'ch_data'
        kernel_size : int
            kernel size of the convolutional layers in the ResNet block
        mode : str {'1D', '2D'}
            Identifies number of feature dimensions, disregarding the channel dimension. (should rather be implemented
            as an int)
        ch_multipliers : Tuple
            Identifies the number of channels in a resolution level. For the default values this would mean:
                - level 0: n_channels = ch_init * 1 = 16
                - level 1: n_channels = ch_init * 2 = 32
                - level 2: n_channels = ch_init * 4 = 64
            This means, each entry of the Tuple corresponds to one resolution layer. After each layer, the feature
            dimensions are halved. Consider this, when deciding on the number of resolution levels
        num_res_blocks : int
            Number of ResNet blocks in one resolution level
        dropout : float
            percentage of dropped samples in Dropout layer during training
        norm_type : str {'batch', 'group', 'instance'}
            type of normalization layer that is used
        device : torch.device or str {'cpu', 'cuda', 'cuda:id', ...}
            Device on which the NN is working.
        """

        super().__init__()
        self.ch_data = ch_data
        self.device = utils.set_device(device)
        self.ch_out = utils.default(ch_out, ch_data)

        num_norm_groups = min(ch_init, 8)

        # initial convolutional layer, converting the number of channels to 'ch_init'
        if mode == '1D':
            self.init_conv = nn.Conv1d(ch_data, ch_init, kernel_size=kernel_size, padding='same',
                                       padding_mode='replicate', device=self.device)
        elif mode == '2D':
            self.init_conv = nn.Conv2d(ch_data, ch_init, kernel_size=kernel_size, padding='same',
                                       padding_mode='replicate', device=self.device)
        else:
            raise ValueError(f'Upsample mode {self.mode} is not supported.')

        # Time embedding related functionalities, computing the base time embedding
        self.time_embedding_func = lambda t: get_positional_embedding(t, ch_init)
        # The time embedding dimension can also be set different. It should not be too small
        self.time_emb_dim = ch_init * 4
        self.time_mlp = nn.Sequential(nn.Linear(ch_init, self.time_emb_dim, device=self.device),
                                      nn.SiLU(),
                                      nn.Linear(self.time_emb_dim, self.time_emb_dim, device=self.device))

        # identify channel dimensions in each resolution level
        ch_dims = [ch_init * multiplier for multiplier in ch_multipliers]
        num_resolution_levels = len(ch_dims)

        # Downsampling blocks. Since PyTorch requires to know the number of channels in each layer at instantiation of
        # the model, logic in constructing the downsampling and upsampling modules is quite complex
        self.downs = nn.ModuleList()
        for res_step in range(num_resolution_levels):
            for block in range(num_res_blocks):
                if res_step == 0 and block == 0:
                    # the first block of the first resolution level gets the tensor after the initial convolution
                    ch_in = ch_init
                elif res_step != 0 and block == 0:
                    # The first block in each resolution level receives an Input Tensor that still has the channel
                    # dimension of the previous resolution level
                    ch_in = ch_dims[res_step - 1]
                else:
                    ch_in = ch_dims[res_step]
                self.downs.append(
                    ResNetBlock(ch_in, ch_dims[res_step], time_emb_dim=self.time_emb_dim, kernel_size=kernel_size,
                                mode=mode, dropout=dropout, norm_type=norm_type, num_groups=num_norm_groups,
                                device=self.device))
            # Append a downsampling layer after each resolution level but the last one
            if res_step != num_resolution_levels - 1:
                self.downs.append(Downsample(ch_dims[res_step], mode=mode, with_conv=False, device=self.device))

        # Middle ResNet block
        self.mid_resnet = ResNetBlock(ch_dims[-1], time_emb_dim=self.time_emb_dim, kernel_size=kernel_size, mode=mode,
                                      dropout=dropout, norm_type=norm_type, num_groups=num_norm_groups,
                                      device=self.device)

        # Upsampling blocks
        self.ups = nn.ModuleList()
        for res_step in reversed(range(num_resolution_levels)):
            # The upsampling stream has one ResNet block more than the downsampling stream in each resolution level
            for block in range(num_res_blocks + 1):
                if res_step == num_resolution_levels - 1:
                    # because of the residual connections, the number of input channels of the different ResNet blocks
                    # differs a lot depending on the resolution level and the current block. Look at the block diagram
                    # provided in the thesis for a better understanding
                    if block == num_res_blocks and num_resolution_levels > 1:
                        ch_in = ch_dims[res_step - 1] + ch_dims[res_step]
                    else:
                        ch_in = 2 * ch_dims[res_step]
                elif res_step == 0:
                    if block == 0 and num_resolution_levels > 1:
                        ch_in = ch_dims[res_step + 1] + ch_dims[res_step]
                    else:
                        ch_in = 2 * ch_dims[res_step]
                else:
                    if block == 0:
                        ch_in = ch_dims[res_step + 1] + ch_dims[res_step]
                    elif block == num_res_blocks:
                        ch_in = ch_dims[res_step - 1] + ch_dims[res_step]
                    else:
                        ch_in = 2 * ch_dims[res_step]
                self.ups.append(
                    ResNetBlock(ch_in, ch_dims[res_step], time_emb_dim=self.time_emb_dim, kernel_size=kernel_size,
                                mode=mode, dropout=dropout, norm_type=norm_type, num_groups=num_norm_groups,
                                device=self.device))
            # Add an Upsampling layer at each resolution level but the first one
            if res_step != 0:
                self.ups.append(Upsample(ch_dims[res_step], with_conv=False, mode=mode, device=self.device))

        # final normalization layer
        norm_final = get_normalization_layer(norm_type=norm_type, num_channels=ch_init, num_groups=num_norm_groups,
                                             device=self.device)

        # final convolutional layer, converting the number of channels to 'self.ch_out'
        if mode == '1D':
            final_conv = nn.Conv1d(ch_init, self.ch_out, kernel_size=kernel_size, padding='same',
                                   padding_mode='replicate', device=self.device)
        elif mode == '2D':
            final_conv = nn.Conv2d(ch_init, self.ch_out, kernel_size=kernel_size, padding='same',
                                   padding_mode='replicate', device=self.device)
        else:
            raise ValueError(f'Upsample mode {self.mode} is not supported.')

        self.final_block = nn.Sequential(
            norm_final,
            nn.SiLU(),
            final_conv,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass of the complete U-Net

        Parameters
        ----------
        x : Tensor of shape [batch_size, ch_data, dim1, dim2, ...]
            batched Input Tensor that is going to be processed by the U-Net
        t : Tensor of shape [batch_size]
            one DM timestep for each data sample.

        Returns
        -------
        x : Tensor of shape [batch_size, ch_out, dim1, dim2, ...]
            batched Output Tensor of the U-Net
        """

        # compute the time embedding for all timesteps
        t_emb = self.time_mlp(self.time_embedding_func(t))

        h = self.init_conv(x)

        # store intermediate feature Tensors for residual connections between down- and upsampling stream
        residuals = [h]

        for down in self.downs:
            if isinstance(down, ResNetBlock):
                # ResNet block computation
                h = down(h, t_emb)
                residuals.append(h)
            else:
                # Downsampling layer
                h = down(h)
                residuals.append(h)

        h = self.mid_resnet(h, t_emb)

        for up in self.ups:
            if isinstance(up, ResNetBlock):
                # ResNet block computation, where the channels of current and residual features are stacked
                h = up(torch.cat([h, residuals.pop()], dim=1), t_emb)
            else:
                # Upsampling layer
                h = up(h)

        h = self.final_block(h)
        assert h.shape == x.shape
        return h
