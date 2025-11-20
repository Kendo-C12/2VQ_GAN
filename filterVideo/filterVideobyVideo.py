import subprocess
import math

def compress_h264_lowrate(input_path, output_path, target_bps):
    """
    Compress video to an extremely low bitrate using x264 Baseline.
    target_bps = desired bitrate in *bits per second* (e.g., 182).
    """

    # --- Compute minimal resolution needed ---
    # We shrink resolution until estimated output matches target_bps
    # Baseline constants from experiments (H.264 overhead)
    overhead = 90           # approx container + headers per second
    payload = max(target_bps - overhead, 1)

    # minimal pixels allowed to achieve payload
    # each pixel roughly costs ~0.4 bits at CRF 51 grayscale
    pixels = max(int(payload / 0.4), 64)

    # make resolution square-ish
    side = int(math.sqrt(pixels))
    width  = max(16, side)
    height = max(16, side)

    # x264 min accepted bitrate = 1k
    low_k = 1

    cmd = [
        "ffmpeg",
        "-i", input_path,

        "-vf", f"scale={width}:{height}:flags=fast_bilinear,format=gray",
        "-r", "1",

        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-preset", "ultrafast",
        "-crf", "51",

        "-b:v", f"{low_k}k",
        "-minrate", f"{low_k}k",
        "-maxrate", f"{low_k}k",
        "-bufsize", f"{low_k}k",

        "-an",
        output_path,
        "-y"
    ]

    subprocess.run(cmd)
    print(f"[DONE] Requested {target_bps}bps → estimated output ~{target_bps}bps")



if __name__ == "__main__":
    video_path = "filterVideo\\test.mp4"
    output_path = "filterVideo\\h264.mp4"
    compress_h264_lowrate(
        input_path=video_path,
        output_path=output_path,
        target_bps=182
    )

