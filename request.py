import requests

url = 'http://127.0.0.1:5000/predict'

# The paths to your images
image1_path = '/Users/ishnoorsingh/Code/mlops_assignment/image1.png'
image2_path = '/Users/ishnoorsingh/Code/mlops_assignment/image2.png'

# Opening the image files in binary read mode
with open(image1_path, 'rb') as img1, open(image2_path, 'rb') as img2:
    files = [
        ('images', ('image1.png', img1, 'image/png')),
        ('images', ('image2.png', img2, 'image/png'))
    ]
    response = requests.post(url, files=files)

# Checking the response
print(response.json())
