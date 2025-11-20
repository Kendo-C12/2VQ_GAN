# import av
# import numpy as np
# import os

# def compress_h264_pyav(input_path, output_path, width, height, fps, bitrate):
#     input_container = av.open(input_path)
#     output_container = av.open(output_path, mode='w', format='mp4')

#     # Video stream output
#     stream = output_container.add_stream('libx264', rate=fps) # h264 codec
#     stream.width = width
#     stream.height = height
#     stream.pix_fmt = 'yuv420p'
#     stream.options = {
#         'preset': 'veryfast',
#         'profile': 'baseline',
#         'tune': 'zerolatency',
#         'crf': '35',
#         'b': f'{bitrate}',
#     }

#     for frame in input_container.decode(video=0):
#         frame = frame.reformat(width, height)
#         packets = stream.encode(frame)
#         if packets is list():
#             for packet in packets:
#                 output_container.mux(packet)
#         else:    
#             output_container.mux(packets)

#     # Flush encoder
#     packets = stream.encode(None)
#     if packets is list():
#         for packet in packets:
#             output_container.mux(packet)
#     else:    
#         output_container.mux(packets)

#     output_container.close()
#     input_container.close()

import av
import os

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

    for frame in input_container.decode(video=0):
        frame = frame.reformat(width, height)
        for packet in stream.encode(frame):   # <-- must use loop
            output_container.mux(packet)

    # flush remaining packets
    for packet in stream.encode():
        output_container.mux(packet)

    output_container.close()
    input_container.close()


# Example
if __name__ == "__main__":

    print("encoders:", av.codec.codecs_available)

    video_path = os.path.join("filterVideo", "dataset", "blur.mp4")
    output_path = os.path.join("filterVideo", "result", "h264_lib.mp4")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    compress_h264_pyav(
        input_path=video_path,
        output_path=output_path,
        width=80,
        height=60,
        fps=30,
        bitrate="5000"  # in bits per second
    )