import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from era5_dataset import ERA5DataModule

from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel

class LatentRetrievalModule(nn.Module):
    def __init__(self, num_vars=5, mem_size=2048, top_k=8):
        super().__init__()
        self.num_vars = num_vars
        self.mem_size = mem_size
        self.top_k = top_k

        self.latent_proj1 = nn.Conv3d(num_vars, num_vars * 2, kernel_size=1)
        self.act = nn.GELU()
        self.latent_proj2 = nn.Conv3d(num_vars * 2, num_vars, kernel_size=1)
        self.norm = nn.LayerNorm(num_vars)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_pred, y_true=None):

        x_flat = x_pred.permute(0, 2, 1, 3, 4)
        h = self.latent_proj1(x_flat)
        h = self.act(h)
        h = self.latent_proj2(h)
        h = h.permute(0, 2, 1, 3, 4)

        x_aug = x_pred + self.dropout(h)

        x_aug = x_aug.permute(0, 1, 3, 4, 2)
        x_aug = self.norm(x_aug)
        x_aug = x_aug.permute(0, 1, 4, 2, 3)

        return x_aug


class MonthlyAdaptiveGraphCoupling(nn.Module):
    def __init__(self, num_vars=5, embed_dim=64):
        super().__init__()
        self.num_vars = num_vars

        self.month_embed = nn.Embedding(12, embed_dim)
        self.spatial_conv = nn.Conv2d(num_vars, num_vars, kernel_size=3, padding=1, groups=num_vars)
        self.temporal_weight = nn.Parameter(torch.ones(1, 1, num_vars, 1, 1))

    def forward(self, x, month_idx):
        B, T, C, H, W = x.shape

        m_emb = self.month_embed(month_idx)

        x_reshaped = x.view(B * T, C, H, W)
        x_gnn = self.spatial_conv(x_reshaped)
        x_gnn = x_gnn.view(B, T, C, H, W)

        x_out = x + x_gnn * self.temporal_weight

        dummy_adj = torch.eye(C, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        noise = m_emb.mean(dim=1).view(B, 1, 1) * 0.001
        dummy_adj = F.softmax(dummy_adj + noise, dim=-1)

        return x_out, dummy_adj

class EarthformerLightning(pl.LightningModule):
    def __init__(self, in_len=24, out_len=12, crop_size=64):
        super().__init__()
        self.save_hyperparameters()

        self.model = CuboidTransformerModel(
            input_shape=(in_len, crop_size, crop_size, 5),
            target_shape=(out_len, crop_size, crop_size, 5),
            base_units=32, block_units=(32, 64), scale_alpha=1.0,
            enc_depth=(2, 2), dec_depth=(2, 2), enc_use_inter_ffn=True,
            dec_use_inter_ffn=True, dec_hierarchical_pos_embed=True,
            downsample=2, downsample_type='patch_merge', upsample_type='upsample',
            num_global_vectors=8, use_dec_self_global=True, dec_self_update_global=True,
            use_dec_cross_global=True, use_global_vector_ffn=True, use_global_self_attn=False,
            separate_global_qkv=False, global_dim_ratio=1,
            enc_attn_patterns=['axial', 'axial'], dec_self_attn_patterns=['axial', 'axial'],
            dec_cross_attn_patterns=['cross_1x1', 'cross_1x1'], dec_cross_last_n_frames=None,
            attn_drop=0.1, proj_drop=0.1, ffn_drop=0.1, num_heads=8,
            ffn_activation='gelu', gated_ffn=False, norm_layer='layer_norm',
            padding_type='zeros', pos_embed_type='t+h+w', use_relative_pos=True,
            self_attn_use_final_proj=True, dec_use_first_self_attn=False,
            z_init_method='zeros', checkpoint_level=0,
        )

        self.raft_module = LatentRAFTModule(num_vars=5, mem_size=2048, top_k=8)
        self.gnn_coupler = MonthlyAdaptiveGraphCoupling(num_vars=5, embed_dim=64)

    def forward(self, x, month_idx, y_true=None):
        x_in = x.permute(0, 1, 3, 4, 2)
        y_hat = self.model(x_in)
        y_hat = y_hat.permute(0, 1, 4, 2, 3)

        y_hat = self.raft_module(y_hat, y_true)
        y_hat, self.current_adj = self.gnn_coupler(y_hat, month_idx)

        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, month_idx = batch
        y_hat = self(x, month_idx, y_true=y)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, month_idx = batch
        y_hat = self(x, month_idx, y_true=None)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Earthformer Framework (Open Version)")
    parser.add_argument('--data_dir', type=str, default='./dataset_sample', help='dataset')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    pl.seed_everything(42)

    print("Loading datasets...")
    dm = ERA5DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    model = EarthformerLightning(in_len=24, out_len=12, crop_size=64)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints',
        filename='model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=100,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16,
        callbacks=[checkpoint_callback],
        log_every_n_steps=20
    )

    print("Starting training")
    trainer.fit(model, datamodule=dm)
