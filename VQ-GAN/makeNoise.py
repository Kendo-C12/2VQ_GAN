import numpy as np
from PIL import Image, ImageFilter
import io
import random

# 1️⃣ Compress image for LoRa transmission
def compress_image(image, target_size_kb=8):
    """
    Compress image to small size suitable for LoRa (default ~8 KB).
    Takes PIL image as input, not path.
    """
    # Resize to small resolution (LoRa can't handle big packets)
    img = image.resize((160, 120))  # small thumbnail

    # Save with reduced quality until under target size
    quality = 85
    while True:
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        size_kb = len(buffer.getvalue()) / 1024
        if size_kb <= target_size_kb or quality <= 20:
            break
        quality -= 5
    buffer.seek(0)
    return Image.open(buffer)


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
def simulate_lora_transmission(image_path):
    """
    Simulate camera capture → compression → transmission over 30 km LoRa.
    """
    print("[1] Loading image...")
    original = Image.open(image_path).convert('RGB')

    print("[2] Adding camera sensor noise...")
    camera_noisy = add_camera_noise(original)

    print("[3] Compressing image for LoRa...")
    compressed = compress_image(camera_noisy)

    # print("[4] Simulating LoRa transmission (distance = 30 km)...")
    # transmitted = add_transmission_noise(compressed, distance=30_000)

    print("[✔] Simulation complete.")
    return compressed


# Example usage:
if __name__ == "__main__":
    input_image_path = "test.jpg"
    input_image = Image.open(input_image_path)
    result = simulate_lora_transmission(input_image_path)
    result.show()  # Display simulated received image
    result.save("lora_received.jpg")

    print(f"Original size: {input_image.size}\nAfter receive size: {result.size}")
