import os
from PIL import Image
import numpy as np

def generate_random_image(image_size=(64, 64)):
    random_image_array = np.random.randint(0, 256, (image_size[0], image_size[1], 3), dtype=np.uint8)
    random_image = Image.fromarray(random_image_array)
    
    return random_image

def save_random_images(num_images=10, save_dir='random_images', image_size=(64, 64)):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Image saving started..................................")
    for i in range(num_images):
        random_image = generate_random_image(image_size)
        image_path = os.path.join(save_dir, f'random_image_{i}.png')
        random_image.save(image_path)
        # print(f"Image saved at {image_path}")

    print("Image saving completed..................................")

if __name__ == '__main__':
    save_random_images(num_images=10, image_size=(64, 64))
