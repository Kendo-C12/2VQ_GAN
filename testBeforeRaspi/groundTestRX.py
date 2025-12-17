import serial
import time
import av
import cv2
import traceback
import numpy as np

# Open COM port
ser = serial.Serial(
    port='COM3',        # change to your port: COM3, COM4, /dev/ttyUSB0, etc.
    baudrate=115200,
    timeout=1           # seconds
)

fps = 12
start = time.time()
frame_idx = 0
MAX_BYTE = 260
packetLeftOld = 0
overallPacket = b""

codecDE = av.codec.CodecContext.create("h264", "r")

codecDE.pix_fmt = "yuv420p"
codecDE.width = 640
codecDE.height = 480

codecDE.bit_rate = 100 * 1000
codecDE.framerate = 12
codecDE.thread_count = 4
codecDE.profile = "baseline"

codecDE.options = {
    "preset": "ultrafast",
    "tune": "zerolatency",
    "keyint": str(fps),        # max I-frame interval
    "min-keyint": str(fps),    # force exactly 12
    "scenecut": "0",       # prevent auto keyframes
    "bframes": "0"
}

codecDE.open()

def decode_h264(byte):
    h264byte = av.packet.Packet(byte)
    print("Before decode type: ",type(h264byte))
    frames = codecDE.decode(h264byte)
    print("There are:",len(frames)," frames")
    for frame in frames:
        return frame.to_ndarray(format="rgb24")

if __name__ == "__main__":
    try:
        while True:
            line = ser.read(MAX_BYTE)

            comma_index = line.find(b',')

            head = line[:comma_index].decode()  # 'packet'
            frameNUM = np.uint8(line[comma_index + 1])
            packet = line[comma_index + 2:]      # first byte after comma
            packetLeft = np.uint8(line[-1])

            if packetLeftOld - packetLeft != 1 or (frame_idx % 256 != frameNUM and packetLeftOld != 0):
                missingPac = packetLeftOld - packetLeft
                if packetLeft > packetLeftOld:
                    missingPac = packetLeftOld
                print("FRAME IDX: ",frame_idx,"\t\tMISSING: ",str(missingPac),"packet")


            if (packetLeftOld == 0) or (frame_idx % 256 != frameNUM):
                try:
                    frame = decode_h264(line)
                    cv2.imshow("Frame",frame)

                    now = time.time()-start
                    print("FRAME IDX: ",frame_idx)
                    print("\tsecond: ",now)
                    print("\texpect frame idx: ",now * fps)
                    print("\tDELAY: ",now - (frame_idx/fps)," SECOND")
                except Exception as e:
                    print("ERROR: ",e)
                    traceback.print_exc()
                packet = b''

            overallPacket += packet

            packetLeftOld = packetLeft
            frame_idx += frame_idx - frameNUM

    except KeyboardInterrupt:
        cv2.destroyAllWindows()