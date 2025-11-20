"""
Automatic Video Dataset Downloader
Downloads videos from various sources for training VQ-VAE2 model
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import requests
import subprocess
import zipfile
import tarfile
from urllib.parse import urlparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoDatasetDownloader:
    """Download and prepare video datasets from various sources"""
    
    def __init__(self, output_dir="datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if yt-dlp is available (better than youtube-dl)
        self.has_ytdlp = self._check_ytdlp()
    
    def _check_ytdlp(self):
        """Check if yt-dlp is installed"""
        try:
            subprocess.run(['yt-dlp', '--version'], 
                         capture_output=True, check=True)
            logger.info("✓ yt-dlp is available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("yt-dlp not found. Install with: pip install yt-dlp")
            return False
    
    def download_file(self, url, output_path, desc="Downloading"):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        logger.info(f"Downloaded: {output_path}")
    
    def extract_archive(self, archive_path, extract_to):
        """Extract zip or tar archive"""
        archive_path = Path(archive_path)
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting {archive_path.name}...")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.bz2', '.xz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
        
        logger.info(f"Extracted to: {extract_to}")
    
    # =========================================================================
    # PUBLIC DATASETS
    # =========================================================================
    
    def download_ucf101(self):
        """
        Download UCF-101 Action Recognition Dataset
        13,320 videos from 101 action categories
        """
        dataset_name = "ucf101"
        dataset_dir = self.output_dir # / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading UCF-101 Dataset...")
        
        # UCF-101 video archive
        video_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
        
        logger.info("Note: UCF-101 is in RAR format. You may need to extract manually.")
        logger.info(f"Download URL: {video_url}")
        logger.info(f"Save to: {dataset_dir}")
        
        # Alternative: Use smaller subset from Kaggle
        logger.info("\nAlternative: Download from Kaggle (requires kaggle API):")
        logger.info("  kaggle datasets download -d pevogam/ucf101")
        
        return dataset_dir
    
    def download_kinetics_sample(self, num_videos=100):
        """
        Download sample from Kinetics-400 dataset
        Note: Full Kinetics requires special download script
        """
        dataset_name = "kinetics_sample"
        dataset_dir = self.output_dir # / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading {num_videos} sample videos from Kinetics...")
        logger.info("Note: Full Kinetics-400 requires the official download script")
        logger.info("Visit: https://github.com/cvdfoundation/kinetics-dataset")
        
        return dataset_dir
    
    def download_moving_mnist(self):
        """
        Download Moving MNIST dataset (simple video dataset for testing)
        10,000 sequences of 20 frames each
        """
        dataset_name = "moving_mnist"
        dataset_dir = self.output_dir # / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading Moving MNIST...")
        
        url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
        output_file = dataset_dir / "mnist_test_seq.npy"
        
        if not output_file.exists():
            self.download_file(url, output_file, "Moving MNIST")
            logger.info("✓ Moving MNIST downloaded")
        else:
            logger.info("✓ Moving MNIST already exists")
        
        # Convert to video files
        self._convert_moving_mnist_to_videos(output_file, dataset_dir)
        
        return dataset_dir
    
    def _convert_moving_mnist_to_videos(self, npy_file, output_dir):
        """Convert Moving MNIST numpy to video files"""
        try:
            import numpy as np
            import cv2
            
            logger.info("Converting Moving MNIST to video files...")
            data = np.load(npy_file)  # Shape: (20, 10000, 64, 64)
            
            video_dir = output_dir # / "videos"
            video_dir.mkdir(exist_ok=True)
            
            # Convert first 100 sequences
            for i in range(min(100, data.shape[1])):
                video_path = video_dir / f"moving_mnist_{i:04d}.mp4"
                
                if not video_path.exists():
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 64), False)
                    
                    for t in range(data.shape[0]):
                        frame = data[t, i]
                        out.write(frame.astype(np.uint8))
                    
                    out.release()
            
            logger.info(f"✓ Converted {min(100, data.shape[1])} videos to {video_dir}")
            
        except ImportError:
            logger.warning("numpy or opencv not available for conversion")
    
    def download_hmdb51(self):
        """
        Download HMDB-51 Action Recognition Dataset
        ~7,000 videos from 51 action categories
        """
        dataset_name = "hmdb51"
        dataset_dir = self.output_dir # / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading HMDB-51 Dataset...")
        
        url = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
        
        logger.info("Note: HMDB-51 is in RAR format.")
        logger.info(f"Download URL: {url}")
        logger.info(f"Save to: {dataset_dir}")
        
        return dataset_dir
    
    # =========================================================================
    # YOUTUBE DOWNLOAD
    # =========================================================================
    
    def download_from_youtube(self, video_url, output_name=None):
        """
        Download single video from YouTube
        
        Args:
            video_url: YouTube video URL
            output_name: Custom output filename (without extension)
        """
        if not self.has_ytdlp:
            logger.error("yt-dlp is required. Install: pip install yt-dlp")
            return None
        
        youtube_dir = self.output_dir # / "youtube_videos"
        youtube_dir.mkdir(parents=True, exist_ok=True)
        
        if output_name:
            output_template = str(youtube_dir / f"{output_name}.%(ext)s")
        else:
            output_template = str(youtube_dir / "%(title)s.%(ext)s")
        
        cmd = [
            'yt-dlp',
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '--merge-output-format', 'mp4',
            '-o', output_template,
            video_url
        ]
        
        logger.info(f"Downloading from YouTube: {video_url}")
        
        try:
            subprocess.run(cmd, check=True)
            logger.info("✓ Download complete")
            return youtube_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e}")
            return None
    
    def download_youtube_playlist(self, playlist_url, max_videos=None):
        """
        Download videos from YouTube playlist
        
        Args:
            playlist_url: YouTube playlist URL
            max_videos: Maximum number of videos to download
        """
        if not self.has_ytdlp:
            logger.error("yt-dlp is required. Install: pip install yt-dlp")
            return None
        
        youtube_dir = self.output_dir # / "youtube_playlist"
        youtube_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'yt-dlp',
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '--merge-output-format', 'mp4',
            '-o', str(youtube_dir / '%(playlist_index)s_%(title)s.%(ext)s'),
        ]
        
        if max_videos:
            cmd.extend(['--playlist-end', str(max_videos)])
        
        cmd.append(playlist_url)
        
        logger.info(f"Downloading playlist: {playlist_url}")
        if max_videos:
            logger.info(f"Max videos: {max_videos}")
        
        try:
            subprocess.run(cmd, check=True)
            logger.info("✓ Playlist download complete")
            return youtube_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e}")
            return None
    
    def download_youtube_search(self, query, max_results=10):
        """
        Download videos from YouTube search query
        
        Args:
            query: Search query string
            max_results: Number of videos to download
        """
        if not self.has_ytdlp:
            logger.error("yt-dlp is required. Install: pip install yt-dlp")
            return None
        
        search_dir = self.output_dir # / "youtube_search" / query.replace(" ", "_")
        search_dir.mkdir(parents=True, exist_ok=True)
        
        search_url = f"ytsearch{max_results}:{query}"
        
        cmd = [
            'yt-dlp',
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '--merge-output-format', 'mp4',
            '-o', str(search_dir / '%(title)s.%(ext)s'),
            search_url
        ]
        
        logger.info(f"Searching YouTube for: {query}")
        logger.info(f"Downloading top {max_results} results...")
        
        try:
            subprocess.run(cmd, check=True)
            logger.info("✓ Search download complete")
            return search_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e}")
            return None
    
    # =========================================================================
    # CUSTOM URL LIST
    # =========================================================================
    
    def download_from_url_list(self, url_list_file):
        """
        Download videos from a text file containing URLs (one per line)
        
        Args:
            url_list_file: Path to text file with URLs
        """
        custom_dir = self.output_dir / "custom_urls"
        custom_dir.mkdir(parents=True, exist_ok=True)
        
        with open(url_list_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Found {len(urls)} URLs to download")
        
        for i, url in enumerate(urls, 1):
            logger.info(f"[{i}/{len(urls)}] Downloading: {url}")
            
            # Determine if it's a direct link or YouTube
            if 'youtube.com' in url or 'youtu.be' in url:
                if self.has_ytdlp:
                    output_name = f"video_{i:04d}"
                    self.download_from_youtube(url, output_name)
            else:
                # Direct download
                filename = f"video_{i:04d}.mp4"
                output_path = custom_dir / filename
                try:
                    self.download_file(url, output_path, f"Video {i}/{len(urls)}")
                except Exception as e:
                    logger.error(f"Failed to download {url}: {e}")
        
        return custom_dir
    
    # =========================================================================
    # DATASET PREPARATION
    # =========================================================================
    
    def prepare_dataset(self, source_dir, target_size=(128, 128), 
                       output_format='mp4', max_duration=10):
        """
        Prepare downloaded videos for training:
        - Resize to target size
        - Trim to max duration
        - Convert to consistent format
        
        Args:
            source_dir: Directory with raw videos
            target_size: (width, height) tuple
            output_format: Output video format
            max_duration: Maximum duration in seconds
        """
        source_dir = Path(source_dir)
        prepared_dir = source_dir.parent / f"{source_dir.name}_prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        
        video_files = list(source_dir.glob('*.mp4')) + \
                     list(source_dir.glob('*.avi')) + \
                     list(source_dir.glob('*.mov'))
        
        logger.info(f"Preparing {len(video_files)} videos...")
        logger.info(f"Target size: {target_size}")
        logger.info(f"Max duration: {max_duration}s")
        
        for video_file in tqdm(video_files, desc="Processing videos"):
            output_file = prepared_dir / f"{video_file.stem}.{output_format}"
            
            if output_file.exists():
                continue
            
            # FFmpeg command to resize and trim
            cmd = [
                'ffmpeg',
                '-i', str(video_file),
                '-t', str(max_duration),
                '-vf', f'scale={target_size[0]}:{target_size[1]}',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-y',
                str(output_file)
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to process {video_file.name}: {e}")
        
        logger.info(f"✓ Prepared dataset saved to: {prepared_dir}")
        return prepared_dir


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Download video datasets for VQ-VAE2 training'
    )
    
    parser.add_argument('--output-dir', type=str, default='datasets',
                       help='Output directory for datasets')
    
    subparsers = parser.add_subparsers(dest='command', help='Download command')
    
    # Public datasets
    subparsers.add_parser('ucf101', help='Download UCF-101 dataset')
    subparsers.add_parser('hmdb51', help='Download HMDB-51 dataset')
    subparsers.add_parser('moving-mnist', help='Download Moving MNIST')
    
    # YouTube
    yt_parser = subparsers.add_parser('youtube', help='Download from YouTube URL')
    yt_parser.add_argument('url', type=str, help='YouTube video URL')
    yt_parser.add_argument('--name', type=str, help='Output filename')
    
    playlist_parser = subparsers.add_parser('playlist', help='Download YouTube playlist')
    playlist_parser.add_argument('url', type=str, help='Playlist URL')
    playlist_parser.add_argument('--max', type=int, help='Max videos to download')
    
    search_parser = subparsers.add_parser('search', help='Download from YouTube search')
    search_parser.add_argument('query', type=str, help='Search query')
    search_parser.add_argument('--max', type=int, default=10, help='Max results')
    
    # Custom URLs
    url_list_parser = subparsers.add_parser('url-list', help='Download from URL list file')
    url_list_parser.add_argument('file', type=str, help='Text file with URLs')
    
    # Prepare
    prep_parser = subparsers.add_parser('prepare', help='Prepare videos for training')
    prep_parser.add_argument('source_dir', type=str, help='Source video directory')
    prep_parser.add_argument('--size', type=int, nargs=2, default=[128, 128],
                            help='Target size (width height)')
    prep_parser.add_argument('--duration', type=int, default=10,
                            help='Max duration in seconds')
    
    args = parser.parse_args()
    
    downloader = VideoDatasetDownloader(output_dir=args.output_dir)
    
    if args.command == 'ucf101':
        downloader.download_ucf101()
    
    elif args.command == 'hmdb51':
        downloader.download_hmdb51()
    
    elif args.command == 'moving-mnist':
        downloader.download_moving_mnist()
    
    elif args.command == 'youtube':
        downloader.download_from_youtube(args.url, args.name)
    
    elif args.command == 'playlist':
        downloader.download_youtube_playlist(args.url, args.max)
    
    elif args.command == 'search':
        downloader.download_youtube_search(args.query, args.max)
    
    elif args.command == 'url-list':
        downloader.download_from_url_list(args.file)
    
    elif args.command == 'prepare':
        downloader.prepare_dataset(
            args.source_dir,
            target_size=tuple(args.size),
            max_duration=args.duration
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()