import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import timm


class TransfuserBackbone(nn.Module):
    """ Multi-scale Fusion Transformer for image + Radar feature fusion """
    def __init__(self,
                 img_vert_anchors,
                 img_horz_anchors,
                 radar_vert_anchors,
                 radar_horz_anchors,
                 perception_output_features,
                 use_point_pillars,
                 num_features,
                 radar_seq_len,
                 use_target_point_image,
                 n_head,
                 block_exp,
                 n_layer,
                 seq_len,
                 embd_pdrop,
                 attn_pdrop,
                 resid_pdrop,
                 heatmap_features_channels,
                 image_architecture='resnet34',
                 radar_architecture='resnet18',
                 gpt_linear_layer_init_mean=0.0,
                 gpt_linear_layer_init_std=0.02,
                 gpt_layer_norm_init_weight=1.0):
        super().__init__()
        
        self.img_vert_anchors = img_vert_anchors
        self.img_horz_anchors = img_horz_anchors
        self.radar_vert_anchors = radar_vert_anchors
        self.radar_horz_anchors = radar_horz_anchors
        self.perception_output_features = perception_output_features
        self.use_point_pillars = use_point_pillars
        self.num_features = num_features
        self.radar_seq_len = radar_seq_len
        self.use_target_point_image = use_target_point_image
        self.n_head = n_head
        self.block_exp = block_exp
        self.n_layer = n_layer
        self.seq_len = seq_len
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.heatmap_features_channels = heatmap_features_channels
        self.gpt_linear_layer_init_mean = gpt_linear_layer_init_mean
        self.gpt_linear_layer_init_std = gpt_linear_layer_init_std
        self.gpt_layer_norm_init_weight = gpt_layer_norm_init_weight

        self.avgpool_img = nn.AdaptiveAvgPool2d((self.img_vert_anchors, self.img_horz_anchors))
        self.avgpool_radar = nn.AdaptiveAvgPool2d((self.radar_vert_anchors, self.radar_horz_anchors))

        self.image_encoder = ImageCNN(architecture=image_architecture, normalize=True)

        if self.use_point_pillars:
            in_channels = self.num_features[-1]
        else:
            in_channels = 2 * self.radar_seq_len

        if self.use_target_point_image:
            in_channels += 1

        self.radar_encoder = RadarEncoder(architecture=radar_architecture, in_channels=in_channels)

        self.transformer1 = GPT(
            n_embd=self.image_encoder.features.feature_info[1]['num_chs'],
            n_head=self.n_head,
            block_exp=self.block_exp,
            n_layer=self.n_layer,
            img_vert_anchors=self.img_vert_anchors,
            img_horz_anchors=self.img_horz_anchors,
            radar_vert_anchors=self.radar_vert_anchors,
            radar_horz_anchors=self.radar_horz_anchors,
            seq_len=self.seq_len,
            embd_pdrop=self.embd_pdrop,
            attn_pdrop=self.attn_pdrop,
            resid_pdrop=self.resid_pdrop,
            gpt_linear_layer_init_mean=self.gpt_linear_layer_init_mean,
            gpt_linear_layer_init_std=self.gpt_linear_layer_init_std,
            gpt_layer_norm_init_weight=self.gpt_layer_norm_init_weight
        )

        self.transformer2 = GPT(
            n_embd=self.image_encoder.features.feature_info[2]['num_chs'],
            n_head=self.n_head,
            block_exp=self.block_exp,
            n_layer=self.n_layer,
            img_vert_anchors=self.img_vert_anchors,
            img_horz_anchors=self.img_horz_anchors,
            radar_vert_anchors=self.radar_vert_anchors,
            radar_horz_anchors=self.radar_horz_anchors,
            seq_len=self.seq_len,
            embd_pdrop=self.embd_pdrop,
            attn_pdrop=self.attn_pdrop,
            resid_pdrop=self.resid_pdrop,
            gpt_linear_layer_init_mean=self.gpt_linear_layer_init_mean,
            gpt_linear_layer_init_std=self.gpt_linear_layer_init_std,
            gpt_layer_norm_init_weight=self.gpt_layer_norm_init_weight
        )

        self.transformer3 = GPT(
            n_embd=self.image_encoder.features.feature_info[3]['num_chs'],
            n_head=self.n_head,
            block_exp=self.block_exp,
            n_layer=self.n_layer,
            img_vert_anchors=self.img_vert_anchors,
            img_horz_anchors=self.img_horz_anchors,
            radar_vert_anchors=self.radar_vert_anchors,
            radar_horz_anchors=self.radar_horz_anchors,
            seq_len=self.seq_len,
            embd_pdrop=self.embd_pdrop,
            attn_pdrop=self.attn_pdrop,
            resid_pdrop=self.resid_pdrop,
            gpt_linear_layer_init_mean=self.gpt_linear_layer_init_mean,
            gpt_linear_layer_init_std=self.gpt_linear_layer_init_std,
            gpt_layer_norm_init_weight=self.gpt_layer_norm_init_weight
        )

        self.transformer4 = GPT(
            n_embd=self.image_encoder.features.feature_info[4]['num_chs'],
            n_head=self.n_head,
            block_exp=self.block_exp,
            n_layer=self.n_layer,
            img_vert_anchors=self.img_vert_anchors,
            img_horz_anchors=self.img_horz_anchors,
            radar_vert_anchors=self.radar_vert_anchors,
            radar_horz_anchors=self.radar_horz_anchors,
            seq_len=self.seq_len,
            embd_pdrop=self.embd_pdrop,
            attn_pdrop=self.attn_pdrop,
            resid_pdrop=self.resid_pdrop,
            gpt_linear_layer_init_mean=self.gpt_linear_layer_init_mean,
            gpt_linear_layer_init_std=self.gpt_linear_layer_init_std,
            gpt_layer_norm_init_weight=self.gpt_layer_norm_init_weight
        )

        if self.image_encoder.features.feature_info[4]['num_chs'] != self.perception_output_features:
            self.change_channel_conv_image = nn.Conv2d(
                self.image_encoder.features.feature_info[4]['num_chs'],
                self.perception_output_features,
                (1, 1)
            )
            self.change_channel_conv_radar = nn.Conv2d(
                self.image_encoder.features.feature_info[4]['num_chs'],
                self.perception_output_features,
                (1, 1)
            )
        else:
            self.change_channel_conv_image = nn.Sequential()
            self.change_channel_conv_radar = nn.Sequential()

    def forward(self, image, radar):
        """
        Image + Radar feature fusion using transformers

        Args:
        - image (tensor): input images
        - radar (tensor): input Radar heatmap
        """
        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image

        radar_tensor = radar

        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.act1(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)
        radar_features = self.radar_encoder._model.conv1(radar_tensor)
        radar_features = self.radar_encoder._model.bn1(radar_features)
        radar_features = self.radar_encoder._model.act1(radar_features)
        radar_features = self.radar_encoder._model.maxpool(radar_features)

        image_features = self.image_encoder.features.layer1(image_features)
        radar_features = self.radar_encoder._model.layer1(radar_features)

        # Image fusion at (B, 72, 40, 176)
        # Radar fusion at (B, 72, 64, 64)
        image_embd_layer1 = self.avgpool_img(image_features)
        radar_embd_layer1 = self.avgpool_radar(radar_features)
        image_features_layer1, radar_features_layer1 = self.transformer1(image_embd_layer1, radar_embd_layer1)
        image_features_layer1 = F.interpolate(image_features_layer1, size=(image_features.shape[2], image_features.shape[3]), mode='bilinear', align_corners=False)
        radar_features_layer1 = F.interpolate(radar_features_layer1, size=(radar_features.shape[2], radar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer1
        radar_features = radar_features + radar_features_layer1

        image_features = self.image_encoder.features.layer2(image_features)
        radar_features = self.radar_encoder._model.layer2(radar_features)

        # Image fusion at (B, 216, 20, 88)
        # Radar fusion at (B, 216, 32, 32)
        image_embd_layer2 = self.avgpool_img(image_features)
        radar_embd_layer2 = self.avgpool_radar(radar_features)
        image_features_layer2, radar_features_layer2 = self.transformer2(image_embd_layer2, radar_embd_layer2)
        image_features_layer2 = F.interpolate(image_features_layer2, size=(image_features.shape[2], image_features.shape[3]), mode='bilinear', align_corners=False)
        radar_features_layer2 = F.interpolate(radar_features_layer2, size=(radar_features.shape[2], radar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer2
        radar_features = radar_features + radar_features_layer2

        image_features = self.image_encoder.features.layer3(image_features)
        radar_features = self.radar_encoder._model.layer3(radar_features)

        # Image fusion at (B, 576, 10, 44)
        # Radar fusion at (B, 576, 16, 16)
        image_embd_layer3 = self.avgpool_img(image_features)
        radar_embd_layer3 = self.avgpool_radar(radar_features)
        image_features_layer3, radar_features_layer3 = self.transformer3(image_embd_layer3, radar_embd_layer3)
        image_features_layer3 = F.interpolate(image_features_layer3, size=(image_features.shape[2], image_features.shape[3]), mode='bilinear', align_corners=False)
        radar_features_layer3 = F.interpolate(radar_features_layer3, size=(radar_features.shape[2], radar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer3
        radar_features = radar_features + radar_features_layer3

        image_features = self.image_encoder.features.layer4(image_features)
        radar_features = self.radar_encoder._model.layer4(radar_features)

        # Image fusion at (B, 1512, 5, 22)
        # Radar fusion at (B, 1512, 8, 8)
        image_embd_layer4 = self.avgpool_img(image_features)
        radar_embd_layer4 = self.avgpool_radar(radar_features)
        image_features_layer4, radar_features_layer4 = self.transformer4(image_embd_layer4, radar_embd_layer4)
        image_features_layer4 = F.interpolate(image_features_layer4, size=(image_features.shape[2], image_features.shape[3]), mode='bilinear', align_corners=False)
        radar_features_layer4 = F.interpolate(radar_features_layer4, size=(radar_features.shape[2], radar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer4
        radar_features = radar_features + radar_features_layer4

        # downsamples channels to perception_output_features
        image_features = self.change_channel_conv_image(image_features)
        radar_features = self.change_channel_conv_radar(radar_features)

        image_features = self.image_encoder.features.global_pool(image_features)
        image_features = torch.flatten(image_features, 1)
        radar_features = self.radar_encoder._model.global_pool(radar_features)
        radar_features = torch.flatten(radar_features, 1)

        fused_features = image_features + radar_features

        return fused_features


class GPT(nn.Module):
    """ The full GPT language model, with a context size of block_size """
    def __init__(self, 
                 n_embd, 
                 n_head, 
                 block_exp, 
                 n_layer,
                 img_vert_anchors, 
                 img_horz_anchors,
                 radar_vert_anchors, 
                 radar_horz_anchors,
                 seq_len,
                 embd_pdrop, 
                 attn_pdrop, 
                 resid_pdrop,
                 gpt_linear_layer_init_mean,
                 gpt_linear_layer_init_std,
                 gpt_layer_norm_init_weight):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = 1
        self.img_vert_anchors = img_vert_anchors
        self.img_horz_anchors = img_horz_anchors
        self.radar_vert_anchors = radar_vert_anchors
        self.radar_horz_anchors = radar_horz_anchors

        # positional embedding parameter (learnable), image + radar
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len * img_vert_anchors * img_horz_anchors + 
                                               self.seq_len * radar_vert_anchors * radar_horz_anchors, n_embd))

        self.drop = nn.Dropout(embd_pdrop)

        # transformer blocks
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_exp, attn_pdrop, resid_pdrop) for _ in range(n_layer)
        ])

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.apply(lambda module: self._init_weights(module, gpt_linear_layer_init_mean, 
                                                     gpt_linear_layer_init_std, 
                                                     gpt_layer_norm_init_weight))

    def _init_weights(self, module, mean, std, norm_weight):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=mean, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(norm_weight)

    def forward(self, image_tensor, radar_tensor):
        """
        Args:
            - image_tensor (tensor): B*4*seq_len, C, H, W
            - radar_tensor (tensor): B*seq_len, C, H, W
        """
        bz = radar_tensor.shape[0]
        radar_h, radar_w = radar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        assert self.seq_len == 1
        image_tensor = image_tensor.view(bz, self.seq_len, -1, img_h, img_w).permute(0, 1, 3, 4, 2).contiguous().view(bz, -1, self.n_embd)
        radar_tensor = radar_tensor.view(bz, self.seq_len, -1, radar_h, radar_w).permute(0, 1, 3, 4, 2).contiguous().view(bz, -1, self.n_embd)

        token_embeddings = torch.cat((image_tensor, radar_tensor), dim=1)

        # add (learnable) positional embedding for all tokens
        x = self.drop(self.pos_emb + token_embeddings)  # (B, an * T, C)

        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        x = x.view(bz, self.seq_len * self.img_vert_anchors * self.img_horz_anchors + 
                   self.seq_len * self.radar_vert_anchors * self.radar_horz_anchors, self.n_embd)

        image_tensor_out = x[:, :self.seq_len * self.img_vert_anchors * self.img_horz_anchors, :].contiguous().view(bz * self.seq_len, -1, img_h, img_w)
        radar_tensor_out = x[:, self.seq_len * self.img_vert_anchors * self.img_horz_anchors:, :].contiguous().view(bz * self.seq_len, -1, radar_h, radar_w)

        return image_tensor_out, radar_tensor_out


class ImageCNN(nn.Module):
    """
    Encoder network for image input list

    Args:
        - architecture (string): vision architecture to be used from the timm model library
        - normalize (bool): whether the input images should be normalized
    """
    def __init__(self, architecture, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = timm.create_model(architecture, pretrained=True)
        self.features.fc = None


def normalize_imagenet(x):
    """
    Normalize input images according to ImageNet standards

    Args:
        - x (tensor): input images
    """
    x = x.clone()  # deep copy
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    
    return x


class RadarEncoder(nn.Module):
    """
    Encoder network for Radar input list

    Args:
        - architecture (string): vision architecture to be used from the timm model library
        - in_channels (int): number of input channels
    """
    def __init__(self, architecture, in_channels=2):
        super().__init__()
        self._model = timm.create_model(architecture, pretrained=False)
        self._model.fc = None

        # change the first conv layer to match the number of input channels
        _tmp = self._model.conv1
        use_bias = (_tmp.bias is not None)
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels,
                                      kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=use_bias)
        
        if use_bias:
            self._model.conv1.bias = _tmp.bias

        del _tmp


class SelfAttention(nn.Module):
    """ A vanilla multi-head masked self-attention layer with a projection at the end """
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        return y


class Block(nn.Module):
    """ An unassuming transformer block """
    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x