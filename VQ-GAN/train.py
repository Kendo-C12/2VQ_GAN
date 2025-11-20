# TRAINER SNIPPET — drop into same project where VQVAE2VideoSystem is defined

import sys
import time
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils

from dataset import VideoClipDataset
from improve2VQ_GAN import VQVAE2VideoSystem

from eval import evaluate_model , print_evaluation_report , VideoEvaluator
from vgg import setup_vgg, compute_vgg_loss

def average(lst):
    if len(lst) == 0:
        return 0   # หรือ return None ตามต้องการ
    return sum(lst) / len(lst)

def create_optimizers(system, lr=2e-4):
    """Create separate optimizers for generator and discriminator"""
    # generator params: encoder, vq embeddings (buffers update via EMA), decoder
    gen_params = list(system.encoder.parameters()) + list(system.decoder.parameters())
    disc_params = list(system.disc.parameters())
    optim_G = torch.optim.Adam(gen_params, lr=lr, betas=(0.5, 0.9))
    optim_D = torch.optim.Adam(disc_params, lr=lr, betas=(0.5, 0.9))
    return optim_G, optim_D

def train_step(system, clips, optim_G, optim_D, device,
               lambda_perc=0.1, lambda_gan=0.1,
               scaler: GradScaler = None, train_disc=True):
    """
    Single training step for one batch of clips
    
    Args:
        clips: (B, T, C, H, W) in [0,1]
        train_disc: if False, skip discriminator update (warmup phase)
        scaler: optional AMP GradScaler for mixed precision
    """

    system.train()
    clips = clips.to(device)
    B, T, C, H, W = clips.shape

    total_g_loss = 0.0
    total_d_loss = 0.0

    # -------------------
    # (A) Discriminator update
    # -------------------
    if train_disc:
        optim_D.zero_grad()
        d_loss = 0.0
        enc_hidden = None
        dec_hidden = None
        
        for t in range(T):
            real = clips[:, t]

            # encode -> quantize -> decode
            (z_b_q, idx_b), (z_t_q, idx_t), enc_hidden = system.encode_step(real, enc_hidden)
            fake, dec_hidden = system.decode_step(z_b_q, z_t_q, dec_hidden)
            
            # D predictions
            real_pred = system.disc(real)
            fake_pred = system.disc(fake.detach())

            # MSE-based patchGAN losses
            d_real = F.mse_loss(real_pred, torch.ones_like(real_pred))
            d_fake = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
            d_loss += (d_real + d_fake)

        d_loss = d_loss / T
        
        # backward + step
        if scaler is not None:
            scaler.scale(d_loss).backward()
            scaler.step(optim_D)
        else:
            d_loss.backward()
            optim_D.step()
        total_d_loss = d_loss.item()

    # -------------------
    # (B) Generator update
    # -------------------
    optim_G.zero_grad()
    g_loss = 0.0
    enc_hidden = None
    dec_hidden = None

    recon_frames = []
    for t in range(T):
        real = clips[:, t]
        (z_b_q, idx_b), (z_t_q, idx_t), enc_hidden = system.encode_step(real, enc_hidden)
        fake, dec_hidden = system.decode_step(z_b_q, z_t_q, dec_hidden)
        recon_frames.append(fake)
        
        # reconstruction loss
        rloss = F.l1_loss(fake, real)

        # perceptual / VGG loss
        ploss = compute_vgg_loss(fake, real)

        # adversarial generator loss
        pred = system.disc(fake)
        gloss_gan = F.mse_loss(pred, torch.ones_like(pred))

        # total loss for this frame
        frame_loss = rloss + lambda_perc * ploss + lambda_gan * gloss_gan
        g_loss += frame_loss

    g_loss = g_loss / T

    # backward + step
    if scaler is not None:
        scaler.scale(g_loss).backward()
        scaler.step(optim_G)
        scaler.update()
    else:
        g_loss.backward()
        optim_G.step()

    total_g_loss = g_loss.item()

    return {'generate_loss': total_g_loss, 'discriminator_loss': total_d_loss} , recon_frames

def train_epoch(system, dataloader, optim_G, optim_D, device,
                epoch, lambda_perc=0.1, lambda_gan=0.1,
                warmup_epochs=5):
    """Train for one epoch"""
    scaler = GradScaler() if device.type == 'cuda' else None
    all_metrics = {
        'generate_loss': [], 
        'discriminator_loss': [],
        'psnr_mean': [],
        'psnr_std': [],
        'ssim_mean': [],
        'ssim_std': [],
        'lpips_mean': [],
        'lpips_std': [],
        'l1_loss': [],
        'l2_loss': [],
        'temporal_consistency': []
    }
    keys = all_metrics.keys()
    
    evaluator = VideoEvaluator(device)

    for batch, clips in enumerate(dataloader):
        # warmup: only train generator for warmup_epochs
        train_disc = epoch >= warmup_epochs
        
        stats , recon_frames = train_step(
            system, clips, optim_G, optim_D, device,
            lambda_perc=lambda_perc, 
            lambda_gan=lambda_gan,
            scaler=scaler, 
            train_disc=train_disc
        )
        recon_videos = torch.stack(recon_frames, dim=1)  # (B, T, C, H, W)
        
        summary, _ = evaluator.evaluate_batch(clips, recon_videos)

        for key in summary.keys():
            stats[key] = summary[key]

        for key in keys:
            if key not in stats:
                raise ValueError(f"Missing key {key} in stats")
            all_metrics[key].append(stats[key])
                
        # Print progress every 10 batches
        if (batch + 1) % 10 == 0:
            avg_g = average(all_metrics['generate_loss'])
            avg_d = average(all_metrics['discriminator_loss'])
            status = "(warmup)" if not train_disc else ""
            print(f"  Batch {batch+1}/{len(dataloader)} {status} - G: {avg_g:.4f}, D: {avg_d:.4f}")
    
    for key in keys:
        if key not in stats:
            raise ValueError(f"Missing key {key} in stats")    
        try:
            all_metrics[key] = average(all_metrics[key])
        except Exception as e:
            print(f"Error averaging key {key}: {e}")
    return all_metrics


if __name__ == "__main__":
    
    f = open("VQ-GAN\\output.txt", "w", encoding="utf-8")
    sys.stdout = f

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize VGG for perceptual loss
    setup_vgg(device)

    batch_size = 4
    
    # Dataset
    dataset = VideoClipDataset(
        video_dir="VQ-GAN\\dataset\\train_videos",
        clip_len=8,
        resize=(128, 128)
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    print(f"Dataset: {len(dataset)} clips, {len(loader)} batches")
    
    # Model
    model = VQVAE2VideoSystem().to(device)
    
    # Create separate optimizers for generator and discriminator
    optim_G, optim_D = create_optimizers(model, lr=2e-4)
    
    # Training hyperparameters
    num_epochs = 50
    warmup_epochs = 1
    lambda_perc = 0.1  # perceptual loss weight
    lambda_gan = 0.1   # GAN loss weight
    
    print(f"\nTraining for {num_epochs} epochs (warmup: {warmup_epochs})")
    print(f"Perceptual weight: {lambda_perc}, GAN weight: {lambda_gan}\n")

    train_metrics = []
    test_metrics = []
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        metrics = train_epoch(
            model, loader, optim_G, optim_D, device,
            epoch=epoch,
            lambda_perc=lambda_perc,
            lambda_gan=lambda_gan,
            warmup_epochs=warmup_epochs
        )
        
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        status = "(warmup)" if epoch < warmup_epochs else ""
        print(f"Epoch {epoch+1}/{num_epochs} {status} - "
              f"G Loss: {metrics['g_loss']:.4f}, "
              f"D Loss: {metrics['d_loss']:.4f}, "
              f"Time: {epoch_time:.1f}s")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_G_state': optim_G.state_dict(),
                'optim_D_state': optim_D.state_dict(),
                'metrics': metrics
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pt')
            print(f"  → Saved checkpoint_epoch_{epoch+1}.pt")
        
        # Generate sample reconstruction every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_dataset = next(iter(loader)).to(device)
                metrics = evaluate_model(model, test_dataset, device, batch_size)
                print_evaluation_report(metrics)
                test_metrics.append(metrics)
            model.train()
        
        print()  # blank line between epochs
    
    print("Training complete!")
    torch.save(model.state_dict(), 'final_model.pt')
    print("Saved final_model.pt")



'''

PSNR > 25: Acceptable, > 30: Good
SSIM > 0.85: Acceptable, > 0.90: Good
LPIPS < 0.15: Good perceptual quality
Codebook usage > 70%: Healthy, < 50%: Codebook collapse issue
Temporal consistency < 0.01: Good temporal stability

'''