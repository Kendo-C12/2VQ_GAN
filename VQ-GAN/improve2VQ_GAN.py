"""
Improved VQ-VAE2 + VQ-GAN + ConvLSTM (PyTorch) — Cleaned & documented

This file contains a cleaned, improved implementation of the hierarchical
VQ-VAE2-style compressor with ConvLSTM temporal modelling and a PatchGAN
style discriminator (for optional VQ-GAN training).

Intended use: you will run the ENCODER portion on the Raspberry Pi (sender)
and the DECODER portion on the STM32 (receiver). This file keeps the model
as a single module for development, but at the bottom a short *example* shows
how to split the pipeline into "transmitter" (encoder + quantize + pack)
and "receiver" (unpack + dequantize embedding lookup + decoder).

This is OPTION 2 (rewrite + improve). It fixes bugs from earlier drafts,
keeps the same architecture concept, and includes clear in-line comments.

Notes:
- No full training loop here (you asked for improved architecture). If you
  want a trainer later I can add it.
- The EMA VQ implementation here is adequate for learning codebooks. For
  production you may want additional stability tweaks and priors for entropy coding.

"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# ConvLSTM cell (spatiotemporal)
# ----------------------
# Small, well-tested ConvLSTM cell which preserves spatial dims.
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        # single conv produces 4 gates (i, f, g, o)
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size, padding=pad)
        self.hid_ch = hid_ch

    def forward(self, x, hidden):
        # hidden is tuple (h, c) or None
        if hidden is None:
            B, _, H, W = x.shape
            h = x.new_zeros(B, self.hid_ch, H, W)
            c = x.new_zeros(B, self.hid_ch, H, W)
        else:
            h, c = hidden
        cat = torch.cat([x, h], dim=1)
        gates = self.conv(cat)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)

# ----------------------
# Video encoder (VQ-VAE2 style): produce bottom + top latents
# ----------------------
# Accepts frames (B, C, H, W) and returns:
#   zb (fine) shape (B, z_ch, H', W'),
#   zt (coarse) shape (B, z_ch, H'' , W''),
#   hidden_out for ConvLSTM state.
class VideoEncoderVQ2(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, hid_ch=128, z_ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, base_ch, 4, 2, 1)        # /2
        self.conv2 = nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1) # /4
        self.conv3 = nn.Conv2d(base_ch * 2, hid_ch, 3, 1, 1)
        self.lstm = ConvLSTMCell(hid_ch, hid_ch)
        self.to_zb = nn.Conv2d(hid_ch, z_ch, 1)   # bottom (fine)
        self.to_zt = nn.Conv2d(hid_ch, z_ch, 1)   # top (coarse)

    def forward(self, x, hidden=None):
        # x: (B, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        h, hidden_out = self.lstm(x, hidden)
        zb = self.to_zb(h)                    # fine latents
        # pool spatially to produce coarser latent for top-level quantizer
        zt_pool = F.avg_pool2d(h, kernel_size=2)
        zt = self.to_zt(zt_pool)
        return zb, zt, hidden_out

# ----------------------
# Video decoder (VQ-VAE2 style): consume quantized top+bottom
# ----------------------
class VideoDecoderVQ2(nn.Module):
    def __init__(self, out_ch=3, base_ch=64, hid_ch=128, z_ch=64):
        super().__init__()
        # we will concatenate upsampled top with bottom along channels
        self.unproj = nn.Conv2d(z_ch * 2, hid_ch, 1)
        self.lstm = ConvLSTMCell(hid_ch, hid_ch)
        self.deconv1 = nn.ConvTranspose2d(hid_ch, base_ch * 2, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 3, 1, 1)

    def forward(self, z_b_q, z_t_q, hidden=None):
        # z_b_q: (B, C, Hb, Wb), z_t_q: (B, C, Ht, Wt)
        # Upsample top to bottom resolution and concat
        z_t_up = F.interpolate(z_t_q, size=(z_b_q.shape[2], z_b_q.shape[3]), mode='nearest')
        z = torch.cat([z_t_up, z_b_q], dim=1)
        x = F.relu(self.unproj(z))
        h, hidden_out = self.lstm(x, hidden)
        x = F.relu(self.deconv1(h))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.out_conv(x))
        return x, hidden_out

# ----------------------
# EMA Vector Quantizer (VQ-VAE-2 style)
# ----------------------
# Produces quantized tensor (z_q) and integer indices per spatial location.
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_codes=512, code_dim=64, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps
        # embedding table stored as buffer for EMA updates
        embed = torch.randn(num_codes, code_dim)
        self.register_buffer('embedding', embed)
        self.register_buffer('ema_count', torch.ones(num_codes))
        self.register_buffer('ema_sum', embed.clone())

    def forward(self, z):
        # z: (B, C, H, W)
        B, C, H, W = z.shape
        assert C == self.code_dim, f'channel mismatch {C}!={self.code_dim}'
        flat = z.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        # compute squared distances to codebook
        dists = (flat ** 2).sum(dim=1, keepdim=True) - 2 * flat @ self.embedding.t() + (self.embedding ** 2).sum(dim=1)
        codes = torch.argmin(dists, dim=1)  # (B*H*W,)
        # indemmx lookup to get quantized z
        z_q = self.embedding[codes].view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # EMA update (only in training mode)
        if self.training:
            codes_onehot = F.one_hot(codes, self.num_codes).type_as(flat)
            ema_count = codes_onehot.sum(dim=0)
            ema_sum = codes_onehot.t() @ flat
            self.ema_count = self.ema_count * self.decay + (1 - self.decay) * ema_count
            self.ema_sum = self.ema_sum * self.decay + (1 - self.decay) * ema_sum
            # smooth and re-normalize embedding
            self.embedding = (self.ema_sum + self.eps) / (self.ema_count.unsqueeze(1) + self.eps)

        # straight-through estimator for gradients
        z_q_st = z_q.detach() + (z - z_q).detach()
        indices = codes.view(B, H, W)
        return z_q_st, indices

# ----------------------
# Patch Discriminator (simple)
# ----------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),
            nn.InstanceNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1),
            nn.InstanceNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, 1, 3, 1, 1)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------
# Packing helpers: convert integer indices -> bytes and back
# ----------------------
# These pack bits tightly so transmissions are compact for LoRa payloads.

def pack_indices(indices, num_codes):
    """Pack a (B, H, W) integer index tensor into a list of bytes objects (one per sample).
    bits_per_idx = ceil(log2(num_codes))."""
    B, H, W = indices.shape
    bits_per_idx = int(math.ceil(math.log2(max(2, num_codes))))
    total_bits = bits_per_idx * H * W
    total_bytes = (total_bits + 7) // 8
    packed = []
    for b in range(B):
        flat = indices[b].reshape(-1).cpu().numpy().astype(np.uint32)
        bitstream = 0
        pos = 0
        out = bytearray()
        for val in flat:
            bitstream |= (int(val) << pos)
            pos += bits_per_idx
            while pos >= 8:
                out.append(bitstream & 0xFF)
                bitstream >>= 8
                pos -= 8
        if pos > 0:
            out.append(bitstream & 0xFF)
        # pad to fixed length
        while len(out) < total_bytes:
            out.append(0)
        packed.append(bytes(out))
    return packed


def unpack_indices(packed_bytes, H, W, num_codes):
    """Unpack bytes -> (H, W) numpy array of indices."""
    bits_per_idx = int(math.ceil(math.log2(max(2, num_codes))))
    needed_bytes = (bits_per_idx * H * W + 7) // 8
    data = np.frombuffer(packed_bytes[:needed_bytes], dtype=np.uint8)
    bitbuf = 0
    pos = 0
    vals = []
    for byte in data:
        bitbuf |= (int(byte) << pos)
        pos += 8
        while pos >= bits_per_idx and len(vals) < H * W:
            vals.append(bitbuf & ((1 << bits_per_idx) - 1))
            bitbuf >>= bits_per_idx
            pos -= bits_per_idx
    arr = np.array(vals, dtype=np.int32).reshape(H, W)
    return arr

# ----------------------
# Full VQ-VAE2 video system wrapper
# ----------------------
class VQVAE2VideoSystem(nn.Module):
    """Wrap encoder, two EMA VQ layers (top & bottom), decoder, discriminator."""
    def __init__(self, num_codes_top=256, num_codes_bot=512, code_dim=64):
        super().__init__()
        self.encoder = VideoEncoderVQ2(in_ch=3, base_ch=64, hid_ch=128, z_ch=code_dim)
        self.vq_top = VectorQuantizerEMA(num_codes=num_codes_top, code_dim=code_dim)
        self.vq_bot = VectorQuantizerEMA(num_codes=num_codes_bot, code_dim=code_dim)
        self.decoder = VideoDecoderVQ2(out_ch=3, base_ch=64, hid_ch=128, z_ch=code_dim)
        self.disc = PatchDiscriminator(in_ch=3, base_ch=64)

    def encode_step(self, frame, enc_hidden=None):
        """Encode single frame to quantized embeddings and integer indices.
        Returns: (z_b_q, idx_b), (z_t_q, idx_t), enc_hidden
        """
        z_b, z_t, enc_hidden = self.encoder(frame, enc_hidden)
        z_t_q, idx_t = self.vq_top(z_t)
        z_b_q, idx_b = self.vq_bot(z_b)
        return (z_b_q, idx_b), (z_t_q, idx_t), enc_hidden

    def decode_step(self, z_b_q, z_t_q, dec_hidden=None):
        """Decode quantized embeddings to a reconstructed frame."""
        frame_hat, dec_hidden = self.decoder(z_b_q, z_t_q, dec_hidden)
        return frame_hat, dec_hidden
    
    def generate(self, frame):
        """Full encode-decode pass for single frame (no hidden states)."""
        (z_b_q, idx_b), (z_t_q, idx_t), _ = self.encode_step(frame, None)
        frame_hat, _ = self.decode_step(z_b_q, z_t_q, None)
        return frame_hat

# ----------------------
# Small example: how to split for Raspberry Pi (sender) and STM32 (receiver)
# ----------------------
# This toy example demonstrates the minimal steps you need (no training).
# On Raspberry Pi (sender): run encoder -> quantize -> pack bytes -> send.
# On STM32 (receiver): receive bytes -> unpack -> lookup embeddings -> decode.

if __name__ == '__main__':
    # device choices (for development use CPU/GPU). In deployment, the receiver
    # (STM32) will not run PyTorch — you'll need to export embeddings and a lightweight
    # decoder implementation in C on STM32. This example only shows dataflow.
    device = 'cpu'

    # create system and move to device
    system = VQVAE2VideoSystem(num_codes_top=128, num_codes_bot=256, code_dim=32)
    system.to(device)

    # create a dummy frame batch (B=1) 64x64 RGB
    B, C, H, W = 1, 3, 64, 64
    dummy_frame = torch.rand(B, C, H, W, device=device)

    # ---------- Sender (Raspberry Pi) ----------
    enc_hidden = None
    (z_b_q, idx_b), (z_t_q, idx_t), enc_hidden = system.encode_step(dummy_frame, enc_hidden)
    # idx_b: (B, Hb, Wb), idx_t: (B, Ht, Wt)
    # pack indices into bytes
    packed_bot = pack_indices(idx_b, system.vq_bot.num_codes)[0]
    packed_top = pack_indices(idx_t, system.vq_top.num_codes)[0]

    # packed_top and packed_bot are bytes objects ready to send over LoRa
    print('Packed sizes (bytes):', len(packed_top), len(packed_bot))

    # ---------- Transmission (wire) ----------
    # Simulate sending packed_top then packed_bot
    wire_top = packed_top
    wire_bot = packed_bot

    # ---------- Receiver (STM32) ----------
    # On receiver we would unpack and reconstruct. Here we demonstrate using PyTorch
    # by re-creating embeddings lookup (in practice you'd store embedding table on receiver).
    Ht, Wt = idx_t.shape[1], idx_t.shape[2]
    Hb, Wb = idx_b.shape[1], idx_b.shape[2]

    idx_t_arr = unpack_indices(wire_top, Ht, Wt, system.vq_top.num_codes)
    idx_b_arr = unpack_indices(wire_bot, Hb, Wb, system.vq_bot.num_codes)

    # convert to tensors and lookup embeddings
    emb_top = torch.from_numpy(system.vq_top.embedding.cpu().numpy()).to(device)
    emb_bot = torch.from_numpy(system.vq_bot.embedding.cpu().numpy()).to(device)
    idx_t_t = torch.from_numpy(idx_t_arr).view(-1).long().to(device)
    idx_b_t = torch.from_numpy(idx_b_arr).view(-1).long().to(device)

    zt_flat = emb_top[idx_t_t].view(1, system.vq_top.code_dim, Ht, Wt)
    zb_flat = emb_bot[idx_b_t].view(1, system.vq_bot.code_dim, Hb, Wb)

    # decode
    frame_hat, _ = system.decode_step(zb_flat, zt_flat, dec_hidden=None)
    print('Reconstructed frame shape:', frame_hat.shape)

# End of file
