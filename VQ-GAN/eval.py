from platform import system
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

from vgg import setup_vgg,compute_vgg_loss

class VideoEvaluator:
    """Comprehensive evaluation metrics for video reconstruction"""
    
    def __init__(self, device='cuda'):
        self.device = device
        # Initialize LPIPS model (perceptual similarity)
        try:
            self.lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
            self.use_lpips = True
        except:
            print("LPIPS not available. Install with: pip install lpips")
            self.use_lpips = False
    
    def evaluate_batch(self, real_videos, recon_videos):
        """
        Evaluate a batch of videos
        
        Args:
            real_videos: (B, T, C, H, W) in [0, 1]
            recon_videos: (B, T, C, H, W) in [0, 1]
        
        Returns:
            dict with all metrics
        """
        B, T, C, H, W = real_videos.shape
        
        metrics = {
            'psnr': [],
            'ssim': [],
            'lpips': [],
            'l1_loss': [],
            'l2_loss': [],
            'temporal_consistency': []
        }
        
        # Convert to numpy for PSNR/SSIM (they work on CPU)
        real_np = real_videos.cpu().detach().numpy()
        recon_np = recon_videos.cpu().detach().numpy()
        
        # Per-frame metrics
        for b in range(B):
            for t in range(T):
                real_frame = real_np[b, t].transpose(1, 2, 0)  # (H, W, C)
                recon_frame = recon_np[b, t].transpose(1, 2, 0)
                
                # PSNR
                psnr_val = psnr(real_frame, recon_frame, data_range=1.0)
                metrics['psnr'].append(psnr_val)
                
                # SSIM
                ssim_val = ssim(real_frame, recon_frame, 
                               data_range=1.0, channel_axis=2)
                metrics['ssim'].append(ssim_val)
                
                # LPIPS (perceptual)
                if self.use_lpips:
                    real_tensor = real_videos[b:b+1, t]  # (1, C, H, W)
                    recon_tensor = recon_videos[b:b+1, t]
                    # LPIPS expects [-1, 1] range
                    lpips_val = self.lpips_fn(
                        real_tensor * 2 - 1, 
                        recon_tensor * 2 - 1
                    ).item()
                    metrics['lpips'].append(lpips_val)
        
        # Pixel-level losses (batch-wise)
        l1 = F.l1_loss(recon_videos, real_videos).item()
        l2 = F.mse_loss(recon_videos, real_videos).item()
        metrics['l1_loss'] = l1
        metrics['l2_loss'] = l2
        
        # Temporal consistency (frame-to-frame similarity)
        for b in range(B):
            temp_consistency = []
            for t in range(T - 1):
                real_diff = F.l1_loss(
                    real_videos[b, t], 
                    real_videos[b, t+1]
                ).item()
                recon_diff = F.l1_loss(
                    recon_videos[b, t], 
                    recon_videos[b, t+1]
                ).item()
                # How well does reconstruction preserve temporal changes?
                consistency = abs(real_diff - recon_diff)
                temp_consistency.append(consistency)
            metrics['temporal_consistency'].extend(temp_consistency)
        
        # Average metrics
        summary = {
            'psnr_mean': np.mean(metrics['psnr']),
            'psnr_std': np.std(metrics['psnr']),
            'ssim_mean': np.mean(metrics['ssim']),
            'ssim_std': np.std(metrics['ssim']),
            'l1_loss': metrics['l1_loss'],
            'l2_loss': metrics['l2_loss'],
            'temporal_consistency': np.mean(metrics['temporal_consistency'])
        }
        
        if self.use_lpips and metrics['lpips']:
            summary['lpips_mean'] = np.mean(metrics['lpips'])
            summary['lpips_std'] = np.std(metrics['lpips'])
        
        return summary, metrics


class CodebookAnalyzer:
    """Analyze VQ codebook utilization"""
    
    @staticmethod
    def compute_codebook_usage(indices_list, codebook_size):
        """
        Compute codebook usage statistics
        
        Args:
            indices_list: List of index tensors from VQ forward passes
            codebook_size: Total number of embeddings in codebook
        
        Returns:
            dict with usage statistics
        """
        all_indices = torch.cat(indices_list).flatten()
        unique_indices = torch.unique(all_indices)
        
        usage_percentage = (len(unique_indices) / codebook_size) * 100
        
        # Compute perplexity (measure of codebook diversity)
        counts = torch.bincount(all_indices, minlength=codebook_size).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # Remove zeros
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs))).item()
        
        return {
            'codebook_usage_pct': usage_percentage,
            'unique_codes_used': len(unique_indices),
            'total_codes': codebook_size,
            'perplexity': perplexity
        }


def evaluate_model(model, dataloader, device, num_batches=None,lambda_perc=0.1, lambda_gan=0.1):
    """
    Full evaluation of the model on validation set
    
    Args:
        model: VQVAE2VideoSystem
        dataloader: validation dataloader
        device: torch device
        num_batches: limit evaluation to N batches (None = all)
    
    Returns:
        dict with aggregated metrics
    """
    model.eval()
    evaluator = VideoEvaluator(device)

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
    
    # Track codebook usage
    bottom_indices = []
    top_indices = []
    
    with torch.no_grad():
        for batch, clips in enumerate(dataloader):
            if num_batches and batch >= num_batches:
                break
            
            clips = clips.to(device)
            B, T, C, H, W = clips.shape
            
            # Reconstruct video
            enc_hidden = None
            dec_hidden = None
            recon_frames = []

            g_loss = 0.0
            d_loss = 0.0
            total_d_loss = 0.0
            total_g_loss = 0.0
            
            for t in range(T):
                frame = clips[:, t]
                (z_b_q, idx_b), (z_t_q, idx_t), enc_hidden = model.encode_step(frame, enc_hidden)
                
                frame_hat, dec_hidden = model.decode_step(z_b_q, z_t_q, dec_hidden)
                recon_frames.append(frame_hat)
                
                bottom_indices.append(idx_b)
                top_indices.append(idx_t)
                
                # reconstruction loss
                rloss = F.l1_loss(frame_hat, frame)

                ploss = compute_vgg_loss(frame_hat, frame)

                pred = model.disc(frame_hat)
                gloss_gan = F.mse_loss(pred, torch.ones_like(pred))

                frame_loss = rloss + lambda_perc * ploss + lambda_gan * gloss_gan
                g_loss += frame_loss

                # Discriminator loss
                real_pred = model.disc(frame)
                fake_pred = model.disc(frame_hat.detach())

                d_real = F.mse_loss(real_pred, torch.ones_like(real_pred))
                d_fake = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
                d_loss += (d_real + d_fake)
            d_loss = d_loss / T
            total_d_loss = d_loss.item()
            g_loss = g_loss / T
            total_g_loss = g_loss.item()

            recon_videos = torch.stack(recon_frames, dim=1)  # (B, T, C, H, W)
            
            # Compute metrics
            summary, _ = evaluator.evaluate_batch(clips, recon_videos)
            summary['generate_loss'] = total_g_loss
            summary['discriminator_loss'] = total_d_loss
            for key in keys:
                if key not in summary:
                    raise ValueError(f"Missing key {key} in stats")
                all_metrics[key].append(summary[key])
               
            # Print progress
            if (batch + 1) % 10 == 0:
                print(f"Evaluated {batch+1}/{len(dataloader)} batches...")
    
    # Aggregate results
    final_metrics = {}
    for key in keys:
        if key not in summary:
            raise ValueError(f"Missing key {key} in stats")
        final_metrics[key] = np.mean(all_metrics[key])
        
    # Codebook analysis
    codebook_stats_bottom = CodebookAnalyzer.compute_codebook_usage(
        bottom_indices, model.vq_bot.num_codes
    )
    codebook_stats_top = CodebookAnalyzer.compute_codebook_usage(
        top_indices, model.vq_top.num_codes
    )
    
    final_metrics['codebook_bottom'] = codebook_stats_bottom
    final_metrics['codebook_top'] = codebook_stats_top
    
    return final_metrics


def print_evaluation_report(metrics):
    """Print a nicely formatted evaluation report"""
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    print("\n📊 Reconstruction Quality:")
    print(f"  PSNR:              {metrics['psnr_mean']:.2f} dB")
    print(f"  SSIM:              {metrics['ssim_mean']:.4f}")
    print(f"  L1 Loss:           {metrics['l1_loss']:.6f}")
    
    if 'lpips_mean' in metrics:
        print(f"  LPIPS (perceptual): {metrics['lpips_mean']:.4f}")
    
    print(f"\n🎬 Temporal Consistency: {metrics['temporal_consistency']:.6f}")
    
    print("\n📚 Codebook Usage:")
    print("  Bottom Level:")
    cb_b = metrics['codebook_bottom']
    print(f"    Used: {cb_b['unique_codes_used']}/{cb_b['total_codes']} "
          f"({cb_b['codebook_usage_pct']:.1f}%)")
    print(f"    Perplexity: {cb_b['perplexity']:.2f}")
    
    print("  Top Level:")
    cb_t = metrics['codebook_top']
    print(f"    Used: {cb_t['unique_codes_used']}/{cb_t['total_codes']} "
          f"({cb_t['codebook_usage_pct']:.1f}%)")
    print(f"    Perplexity: {cb_t['perplexity']:.2f}")
    
    print("\n" + "="*60)


# Example usage in training script
if __name__ == "__main__":
    """
    Add this to your training loop:
    
    # After training epoch
    if (epoch + 1) % 5 == 0:
        print("\nRunning evaluation...")
        eval_metrics = evaluate_model(model, val_loader, device, num_batches=50)
        print_evaluation_report(eval_metrics)
    """
    
    # Example of how metrics look
    sample_metrics = {
        'psnr': 28.5,
        'ssim': 0.92,
        'l1_loss': 0.045,
        'lpips': 0.12,
        'temporal_consistency': 0.003,
        'codebook_bottom': {
            'unique_codes_used': 450,
            'total_codes': 512,
            'codebook_usage_pct': 87.9,
            'perplexity': 380.2
        },
        'codebook_top': {
            'unique_codes_used': 210,
            'total_codes': 256,
            'codebook_usage_pct': 82.0,
            'perplexity': 190.5
        }
    }
    
    print_evaluation_report(sample_metrics)