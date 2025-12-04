import av
import os
import matplotlib.pyplot as plt
import io
import zipfile
import base64

def get_zip_base64_size(bytes_data):
    """Return size in KB after zip + base64 encoding."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("video", bytes_data)
    b64_bytes = base64.b64encode(zip_buffer.getvalue())
    return len(b64_bytes) / 1024


def get_video_duration_seconds(video_path):
    container = av.open(video_path)
    duration = float(container.duration) / 1_000_000  # microseconds → seconds
    container.close()
    return duration

def compress_video_settings(input_path, output_path, width, height, fps, bitrate, container, codec, crf=28):
    """
    Compress a video to H.264 with given width, height, fps, and bitrate.
    Returns the size in bytes of the output video.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    input_container = av.open(input_path)
    output_container = av.open(output_path, mode='w', format='mp4')
    
    stream = output_container.add_stream('libx264', rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'
    stream.bit_rate = int(bitrate)
    stream.options = {
        'preset': 'veryfast',
        'profile': 'baseline',
        'tune': 'zerolatency',
        'crf': str(crf)
    }
    
    for frame in input_container.decode(video=0):
        frame = frame.reformat(width, height)
        for packet in stream.encode(frame):
            output_container.mux(packet)
    for packet in stream.encode():
        output_container.mux(packet)
    
    output_container.close()
    input_container.close()
    
    # Return size after zip + base64
    with open(output_path, "rb") as f:
        raw_bytes = f.read()
    return get_zip_base64_size(raw_bytes)

def compress_video_settings(input_path, output_path, width, height, fps, bitrate, container="mp4", codec="libx264", crf=28):
    """
    Compress a video with specified codec and container using PyAV.
    
    Args:
        input_path: path to input video
        output_path: path to save compressed video
        width, height: target resolution
        fps: frames per second
        bitrate: target bitrate in bits per second
        container: output container format (mp4, avi, mpeg, etc.)
        codec: video codec (libx264, mpeg2video, mpeg4, etc.)
        crf: constant rate factor (quality, lower=better)
    
    Returns:
        Path to compressed video
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    input_container = av.open(input_path)
    output_container = av.open(output_path, mode='w', format=container)
    
    stream = output_container.add_stream(codec, rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'
    
    # Only set bitrate for codecs that support it
    if codec in ['libx264', 'mpeg2video', 'mpeg4']:
        stream.bit_rate = int(bitrate)
    
    # Codec-specific options
    if codec == 'libx264':
        stream.options = {
            'preset': 'veryfast',
            'profile': 'baseline',
            'tune': 'zerolatency',
            'crf': str(crf)
        }
    elif codec == 'mpeg2video':
        stream.options = {
            'qscale': str(crf)  # MPEG-2 uses qscale instead of crf
        }
    elif codec == 'mpeg4':
        stream.options = {
            'qscale': str(crf)  # MPEG-4 Part 2
        }
    
    # Encode frames
    for frame in input_container.decode(video=0):
        frame = frame.reformat(width, height)
        for packet in stream.encode(frame):
            output_container.mux(packet)
    
    # Flush remaining packets
    for packet in stream.encode():
        output_container.mux(packet)
    
    input_container.close()
    output_container.close()
    
    # Return size after zip + base64
    with open(output_path, "rb") as f:
        raw_bytes = f.read()
    return get_zip_base64_size(raw_bytes)


def auto_best_compression(input_path, output_dir="best_compressed_videos"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    input_container = av.open(input_path)
    input_stream = input_container.streams.video[0]
    
    orig_width = input_stream.width
    orig_height = input_stream.height
    orig_fps = float(input_stream.average_rate) if input_stream.average_rate else 30
    input_container.close()
    
    # Candidate settings (resolution × bitrate × CRF)
    resolutions = [(orig_width, orig_height), (orig_width//2, orig_height//2), (orig_width//4, orig_height//4)]
    container = ["mp4", "avi", "mpeg"]
    codec = ["libx264", "mpeg2video", "mpeg4"]
    bitrates = [512000, 256000, 128000]  # in bits per second
    crfs = [28, 30, 35]
    
    best_size = float("inf")
    best_file = None
    
    for w,h in resolutions:
        for br in bitrates:
            for crf in crfs:
                output_path = os.path.join(output_dir, f"{base_name}_{w}x{h}_{br}bps_crf{crf}.mp4")
                size_kb = compress_video_settings(input_path, output_path, w, h, orig_fps, br, crf)
                print(f"Tested {w}x{h} {br}bps CRF{crf}: {size_kb:.2f} KB (ZIP+Base64)")
                if size_kb < best_size:
                    best_size = size_kb
                    best_file = output_path
                    
    print(f"Best compression: {best_file} → {best_size:.2f} KB (ZIP+Base64)")
    return best_file

# Example
if __name__ == "__main__":

    # print("encoders:", av.codec.codecs_available)
    width = 80
    height = 60
    fps = 30
    bitrate = "256"  # in bits per second

    video_path = os.path.join("filterVideo", "dataset", "blur.mp4")
    output_path = os.path.join("filterVideo", "result", f"h264_lib_{width}x{height}_{fps}fps_{bitrate}bps.mp4")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sizes = compress_h264_pyav(
                input_path=video_path,
                output_path=output_path,
                width=width,
                height=height,
                fps=fps,
                bitrate=bitrate  # in bits per second
            )
    real_size = sum(sizes)
    print(f"H.264 Compressed video size: {real_size} bytes")

    duration = get_video_duration_seconds("filterVideo/dataset/blur.mp4")
    print("Video duration:", duration, "seconds")
    print("Average bitrate:", real_size*8/duration, "bps")

    bps = []

    i = 0
    while i < len(sizes):
        bps.append(sum(sizes[i:i+fps]))  # bits per second
        i += fps
        print(f"Second {len(bps)}: {bps[-1]*8} bps")

    plt.figure(figsize=(12,4))
    plt.plot(bps)
    plt.title("H.264 Compressed Frame Size per Frame")
    plt.xlabel("Frame Number")
    plt.ylabel("Size (bytes)")
    plt.show()