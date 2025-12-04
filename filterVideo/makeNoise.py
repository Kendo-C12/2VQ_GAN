import numpy as np
from PIL import Image, ImageFilter
import io
import math
import zipfile
import base64

saveSize = {}

def compress_image(img, target_size_kb, format, min_quality=1, max_quality=100):
    """
    Compress a PIL image to approximate a target size in KB using JPEG quality.
    
    Args:
        image: PIL.Image object
        target_size_kb: desired file size in KB
        min_quality: lowest JPEG quality to try
        max_quality: highest JPEG quality to try
    
    Returns:
        PIL.Image compressed
    """

    # Helper: get size after zip + base64

    def get_base64_size(image_bytes):
        # Base64 encode
        b64_bytes = base64.b64encode(image_bytes)
        return len(b64_bytes) / 1024  # KB

    def get_zip_base64_size(image_bytes):
        # Zip in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("img", image_bytes)
        zip_bytes = zip_buffer.getvalue()
        # Base64 encode
        b64_bytes = base64.b64encode(zip_bytes)
        return len(b64_bytes) / 1024  # KB
    
    # Resize to small resolution first (optional)
    
    # Binary search over JPEG quality
    low = min_quality
    high = max_quality
    best_data = None
    
    while low < high:
        mid = math.ceil((low + high) / 2)
        buffer = io.BytesIO()
        img.save(buffer, format=format, quality=mid)
        size_kb = get_zip_base64_size(buffer.getvalue())

        if size_kb > target_size_kb:
            # file too big, reduce quality
            high = mid - 1
        else:
            # file smaller or equal, try higher quality
            low = mid
    
    # Return compressed image using best quality found
    buffer = io.BytesIO()

    print("Final quality for compression:", low)

    img.save(buffer, format, quality=low)

    size_kb = get_zip_base64_size(buffer.getvalue())

    if size_kb > target_size_kb:
        width, height = img.size
        low = 100 / max(width,height)
        high = 100 
        
        while low < high:
            mid = (low + high) / 2
            
            new_w = max(int(width * mid / 100), 1)
            new_h = max(int(height * mid / 100), 1)
            resized = img.resize((new_w, new_h))
            
            buffer = io.BytesIO()
            resized.save(buffer, format=format, quality=1)
            size_kb = get_zip_base64_size(buffer.getvalue())

            if size_kb >= target_size_kb:
                # file too big, reduce quality
                high = mid - 1
            else:
                # file smaller or equal, try higher quality
                low = mid
        
        new_w = max(int(width * low / 100), 1)
        new_h = max(int(height * low / 100), 1)
        resized = img.resize((new_w, new_h))
        
        buffer = io.BytesIO()
        resized.save(buffer, format=format, quality=1)
        size_kb = get_zip_base64_size(buffer.getvalue())

        if size_kb > target_size_kb:
            lowest = 100 / max(width,height)
            new_w = max(int(width * lowest / 100), 1)
            new_h = max(int(height * lowest / 100), 1)
            resized = img.resize((new_w, new_h))

            buffer = io.BytesIO()
            resized.save(buffer, format=format, quality=1)
            size_kb = get_zip_base64_size(buffer.getvalue())
            if size_kb <= target_size_kb:
                raise KeyError(f"Algorithm is wrong got size of {size_kb} KB, target: {target_size_kb} KB")
            else:    
                raise KeyError(f"Could not reach target size. : {size_kb} KB target: {target_size_kb} KB with width {new_w} height {new_h}")
            

    size_kb = get_zip_base64_size(buffer.getvalue())
    print(f"before zip and base64: {len(buffer.getvalue())/1024} base64 only: {size_kb} ")
    print(f"Final compressed size (KB): {size_kb} with width: {new_w} height {new_h}")
    img = Image.open(buffer)
    img.load()   # force load image data

    return img



# 2️⃣ Add random noise from camera sensor
def add_camera_noise(image, noise_level=0.02):
    """
    Simulate random camera noise (sensor/ISO grain).
    noise_level: fraction of intensity variation.
    """
    arr = np.asarray(image).astype(np.float32) / 255.0
    noise = np.random.normal(0, noise_level, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((noisy_arr * 255).astype(np.uint8))


# 3️⃣ Add random noise from LoRa transmission
def add_transmission_noise(image, distance):
    """
    Simulate signal degradation over distance (30 km -> light noise).
    """
    arr = np.asarray(image).astype(np.float32) / 255.0

    base_noise = 0.001
    distance_factor = distance / 30_000
    noise_strength = base_noise + (distance_factor * 0.005)

    noise = np.random.normal(0, noise_strength, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 1)

    noisy_img = Image.fromarray((noisy_arr * 255).astype(np.uint8))
    noisy_img = noisy_img.filter(ImageFilter.GaussianBlur(radius=0.3))
    return noisy_img


# 4️⃣ Combine all steps
def simulate_lora_transmission(image_path,target_size_kb,format):
    """
    Simulate camera capture → compression → transmission over 30 km LoRa.
    """
    print("target_size_kb:", target_size_kb, "format:", format)

    if image_path.__class__ == str:
        # print("[1] Loading image...")
        original = Image.open(image_path).convert('RGB')
    elif image_path.__class__ == Image.Image:
        original = image_path
        # print("[1] Using provided PIL image...")
    else:
        raise ValueError("Input must be a file path or PIL Image.")

    # print("[2] Adding camera sensor noise...")
    camera_noisy = add_camera_noise(original)

    # print("[3] Compressing image for LoRa...")
    compressed = compress_image(camera_noisy, target_size_kb=target_size_kb, format=format)

    # print("[4] Simulating LoRa transmission (distance = 30 km)...")
    # transmitted = add_transmission_noise(compressed, distance=30_000)

    # print("[✔] Simulation complete.")
    return compressed


# Example usage:
if __name__ == "__main__":
    input_image_path = "test.jpg"
    input_image = Image.open(input_image_path)
    result = simulate_lora_transmission(input_image_path)
    result.show()  # Display simulated received image
    result.save("lora_received.jpg")

    print(f"Original size: {input_image.size}\nAfter receive size: {result.size}")
