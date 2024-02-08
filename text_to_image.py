import io
import random
import numpy as np
import pywt
from PIL import Image

def read_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

def write_file(file_path, data):
    with open(file_path, 'w') as file:
        file.write(data)

def encode_data(data, image_path):
    image = Image.open(image_path)
    img_data = np.array(image)
    coeffs = pywt.dwt2(img_data, 'haar')

    data_ar = np.array([ord(i) for i in data])
    max_size = min(len(data_ar), len(coeffs[0][0]))
    coeffs[0][0][:max_size] = coeffs[0][0][:max_size] * (data_ar[:max_size] / 255.0)

    img_data_new = pywt.idwt2(coeffs, 'haar')
    new_image = Image.fromarray((img_data_new).astype(np.uint8))
    new_image.save('encoded_image.png')

def decode_data(image_path):
    image = Image.open(image_path)
    img_data = np.array(image)
    coeffs = pywt.dwt2(img_data, 'haar')

    data_ar = coeffs[0][0]
    data = [chr(int(round(i * 255.0))) for i in data_ar]
    text = ''.join(data)

    return text

# Read the text file
data = read_file('text_file.txt')

# Encode text data into an image
encode_data(data, 'input_image.png')

# Decode text data from the encoded image
encoded_text = decode_data('encoded_image.png')

# Write the decoded text to a file
write_file('decoded_text.txt', encoded_text)

print(f"Original Data: {data}")
print(f"Encoded Text: {encoded_text}")