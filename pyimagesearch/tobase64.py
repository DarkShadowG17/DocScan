import base64
from PIL import Image

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            # Read image file
            img = img_file.read()
            # Encode image to base64
            encoded_image = base64.b64encode(img)
            return encoded_image.decode('utf-8')
    except FileNotFoundError:
        print("Image file not found!")
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None

if __name__ == "__main__":
    image_path = '.\\ayb.jpg'
    base64_string = image_to_base64(image_path)
    if base64_string:
        print("Base64 representation of the image:")
        print(base64_string)
