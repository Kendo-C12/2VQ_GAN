import os
import cv2
import numpy as np
from PIL import Image
from makeNoise import simulate_lora_transmission

# cv2.INTER_LINEAR = Bilinear interpolation
# cv2.INTER_CUBIC = Bicubic interpolation

kbps = 183 / 1000  # kilobits per second for LoRa at SF12, BW125kHz

def process_frame(frame,bpf,format):
    """
    Convert OpenCV frame → PIL image → LoRa simulation → return processed frame as NumPy array.
    """
    # Convert BGR to RGB (OpenCV uses BGR)
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply LoRa simulation (using your existing function, modified for PIL input)
    processed_pil = simulate_lora_transmission(pil_img,bpf,format)
    # Convert back to OpenCV format (BGR)
    processed_frame = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGB2BGR)

    return processed_frame


def process_video(input_path, format, upsample,max_frames=None, fps=30):
    """
    Process an entire video frame by frame using LoRa simulation.
    """
    print(f"🔄 Processing video: {input_path} with format: {format}, upsample: {upsample}, fps: {fps}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Error: Cannot open video file.")
        return

    fps_video = int(cap.get(cv2.CAP_PROP_FPS)) + 1
    bpf = kbps / fps
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    skip_frames = max(0, int(fps_video / fps))
    print(f"ℹ️ Target FPS: {fps}, bpf: {bpf}, Skipping {skip_frames}")

    output_folder = f"filterVideo\\saveVideo\\{upsample}"
    os.makedirs(output_folder, exist_ok=True)  # create folder if missing

    output_folder = f"filterVideo\\saveVideo\\{upsample}\\{format}"
    os.makedirs(output_folder, exist_ok=True)  # create folder if missing

    output_path = f"filterVideo\\saveVideo\\{upsample}\\{format}\\lora_filtered_with_{fps}fps_{bpf}bpf.avi"
    # Video writer
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width, height)
    )

    if not out.isOpened():
        print("❌ Error: Cannot open VideoWriter. Check codec or output path.")
        cap.release()
        return

    frame_count = 0
    while True:
        frame_count += 1

        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break

        if frame_count % (skip_frames + 1) != 0:
            continue

        processed_frame = process_frame(frame,bpf,format)
        if upsample == "bilinear":
            processed_frame = cv2.resize(processed_frame, (width, height), interpolation=cv2.INTER_LINEAR)
        elif upsample == "bicubic":
            processed_frame = cv2.resize(processed_frame, (width, height), interpolation=cv2.INTER_CUBIC)
        elif upsample == "No_Upsample":
            processed_frame = cv2.resize(processed_frame, (width, height))
        else:
            raise ValueError("Upsample method not recognized.")
        out.write(processed_frame)

        if frame_count // (skip_frames + 1) % 10 == 0:
            print(f"Processed {frame_count // (skip_frames + 1)} frames...")



    cap.release()
    out.release()
    print(f"✅ Done. Output saved to {output_path}")


if __name__ == "__main__":
    video_path = "filterVideo\\test.mp4"
    # for upsample in ["No_Upsample","bilinear","bicubic"]:
    for upsample in ["No_Upsample"]:
        # for format in ['JPEG','PNG','WEBP','JPEG2000']:
        for format in ['JPEG2000']:
            for fps in [0.25,0.5,1]:
                try:
                    if os.path.exists(video_path) :
                        process_video(video_path, upsample=upsample, format=format, fps=fps )
                    else:
                        print(f"❌ Error: Video file '{video_path}' does not exist.")
                        files = [f for f in os.listdir(".") if os.path.isfile(f)]
                        print(files)
                except Exception as e:
                    print(f"❌ Error processing video with upsample={upsample}, format={format}, fps={fps}: {e}")
                    
                    # cap = cv2.VideoCapture(video_path)
                    # if not cap.isOpened():
                    #     print("❌ Error: Cannot open video file.")

                    # fps_video = int(cap.get(cv2.CAP_PROP_FPS)) + 1
                    # bpf = kbps / fps
                    # width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    # output_path = f"filterVideo\\saveVideo\\{upsample}\\{format}\\lora_filtered_with_{fps}fps_{bpf}bpf.avi"
                    # if os.path.exists(output_path):
                    #     os.remove(output_path)
                    # continue