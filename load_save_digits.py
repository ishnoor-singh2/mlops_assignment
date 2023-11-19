import os
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the dataset
digits = datasets.load_digits()

# Create a directory to save the images if it doesn't exist
output_dir = 'digits_images'
os.makedirs(output_dir, exist_ok=True)

# Number of examples to save for each digit
examples_per_digit = 3

for digit in range(10):
    digit_indices = [i for i, label in enumerate(digits.target) if label == digit]
    selected_indices = digit_indices[:examples_per_digit]

    for i, index in enumerate(selected_indices):
        image = digits.images[index]
        plt.imshow(image, cmap='gray')
        plt.axis('off')

        # Save the image
        image_filename = os.path.join(output_dir, f'digit_{digit}_example_{i}.png')
        plt.savefig(image_filename, bbox_inches='tight', pad_inches=0)
        print(f'Saved {image_filename}')
