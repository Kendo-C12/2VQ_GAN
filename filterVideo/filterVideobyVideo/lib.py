import av
import os
import matplotlib.pyplot as plt

def get_video_duration_seconds(video_path):
    container = av.open(video_path)
    duration = float(container.duration) / 1_000_000  # microseconds → seconds
    container.close()
    return duration

def compress_h264_pyav(input_path, output_path, width, height, fps, bitrate):
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
        'crf': '35',
    }

    # store size of each encoded frame
    frame_sizes = []

    for frame in input_container.decode(video=0):
        frame = frame.reformat(width, height)
        for packet in stream.encode(frame):
            frame_sizes.append(packet.size)   # <-- record compressed size
            output_container.mux(packet)

    # flush remaining packets
    for packet in stream.encode():
        frame_sizes.append(packet.size)
        output_container.mux(packet)

    output_container.close()
    input_container.close()

    return frame_sizes  # return list of sizes


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