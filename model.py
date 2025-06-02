import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config import EMBED_DIM, HIDDEN_DIM, PATCH_SIZE, OUT_H, OUT_W


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        identity = x
        out = F.relu(self.fc1(x), inplace=True)
        out = self.fc2(out)
        return out + identity


class DecoderHead(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.sal_fc = nn.Linear(hidden_dim, 1)

    def forward(self, tokens):
        B, N, d = tokens.shape
        x_ = self.proj(tokens)
        h_patches = 288 // PATCH_SIZE
        w_patches = 512 // PATCH_SIZE
        x_4d = x_.view(B, h_patches, w_patches, -1).permute(0, 3, 1, 2)
        up = F.interpolate(x_4d, size=(OUT_H, OUT_W), mode='bilinear', align_corners=False)
        up_4d = up.permute(0, 2, 3, 1)
        out = self.sal_fc(up_4d)
        out = out.permute(0, 3, 1, 2)
        return torch.sigmoid(out)


class ComponentNet(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.face_decoder = DecoderHead(embed_dim, hidden_dim)
        self.text_decoder = DecoderHead(embed_dim, hidden_dim)
        self.banner_decoder = DecoderHead(embed_dim, hidden_dim)
        self.base_decoder = DecoderHead(embed_dim, hidden_dim)

        self.in_fc = nn.Linear(4, 128)
        self.res1 = MLP(128)
        self.res2 = MLP(128)
        self.res3 = MLP(128)
        self.res4 = MLP(128)
        self.res5 = MLP(128)
        self.out_fc = nn.Linear(128, 1)

    def forward(self, tokens):
        face_map = self.face_decoder(tokens)
        text_map = self.text_decoder(tokens)
        banner_map = self.banner_decoder(tokens)
        base_map = self.base_decoder(tokens)

        cat_ = torch.cat([face_map, text_map, banner_map, base_map], dim=1)
        B, C, H, W = cat_.shape
        cat_flat = cat_.view(B, C, H * W).permute(0, 2, 1)

        x_ = self.in_fc(cat_flat)
        x_ = F.relu(x_, inplace=True)
        x_ = self.res1(x_)
        x_ = self.res2(x_)
        x_ = self.res3(x_)
        x_ = self.res4(x_)
        x_ = self.res5(x_)
        out = self.out_fc(x_)
        fused_map = out.view(B, 1, H, W)
        fused_map = torch.sigmoid(fused_map)
        return fused_map, (face_map, text_map, banner_map, base_map)


class VITModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("vit_large_patch16_224",
                                          pretrained=True,
                                          in_chans=6)
        # self.backbone.head= nn.Identity()

        self.backbone.patch_embed.img_size = (288, 512)
        self.backbone.patch_embed.grid_size = (288 // PATCH_SIZE, 512 // PATCH_SIZE)
        self.backbone.patch_embed.num_patches = (288 // PATCH_SIZE) * (512 // PATCH_SIZE)

        self.embed_dim = EMBED_DIM
        self.csp = ComponentNet(embed_dim=self.embed_dim, hidden_dim=HIDDEN_DIM)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ = self.backbone.patch_embed(x)

        old_pe = self.backbone.pos_embed
        new_pe = interpolate_pos_embed(self.backbone.patch_embed, old_pe, H, W).to(x.device)
        cls_tok = new_pe[:, 0:1, :]
        patch_pe = new_pe[:, 1:, :]

        x_ = x_ + patch_pe
        cls_tokens = cls_tok.expand(B, -1, -1)
        x_ = torch.cat([cls_tokens, x_], dim=1)

        x_ = self.backbone.pos_drop(x_)
        for blk in self.backbone.blocks:
            x_ = blk(x_)
        x_ = self.backbone.norm(x_)

        tokens = x_[:, 1:, :]
        final_map, comps = self.csp(tokens)
        return final_map, comps


def interpolate_pos_embed(patch_embed, old_pos_embed, H, W):
    cls_tok = old_pos_embed[:, 0:1, :]
    patch_tok = old_pos_embed[:, 1:, :]
    c = patch_tok.shape[2]
    old_size = int(math.sqrt(patch_tok.shape[1]))
    old_2d = patch_tok.view(1, old_size, old_size, c).permute(0, 3, 1, 2)

    new_h = H // 16
    new_w = W // 16
    new_2d = F.interpolate(old_2d, size=(new_h, new_w), mode='bicubic', align_corners=False)
    new_2d = new_2d.permute(0, 2, 3, 1).reshape(1, new_h * new_w, c)
    return torch.cat([cls_tok, new_2d], dim=1)
