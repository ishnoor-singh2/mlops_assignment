import pytest
from app import app
from io import BytesIO

def test_post_predict():
    sample_images = {
        0: '/Users/ishnoorsingh/Code/mlops_assignment/digit_check/digit_0.png',
        1: '/Users/ishnoorsingh/Code/mlops_assignment/digit_check/digit_1.png',
        2:'/Users/ishnoorsingh/Code/mlops_assignment/digit_check/digit_2.png',
        3:'/Users/ishnoorsingh/Code/mlops_assignment/digit_check/digit_3.png',
        4:'/Users/ishnoorsingh/Code/mlops_assignment/digit_check/digit_4.png',
        5:'/Users/ishnoorsingh/Code/mlops_assignment/digit_check/digit_5.png',
        6:'/Users/ishnoorsingh/Code/mlops_assignment/digit_check/digit_6.png',
        7:'/Users/ishnoorsingh/Code/mlops_assignment/digit_check/digit_7.png',
        8:'/Users/ishnoorsingh/Code/mlops_assignment/digit_check/digit_8.png',
        9:'/Users/ishnoorsingh/Code/mlops_assignment/digit_check/digit_9.png'
    }

    for digit, image_path in sample_images.items():
        with open(image_path, 'rb') as img:
            img_bytes = BytesIO(img.read())
            response = app.test_client().post(
                "/predict",
                content_type='multipart/form-data',
                data={'image': (img_bytes, image_path)}
            )

            assert response.status_code == 200  # Checking the status code
            assert response.json.get('predicted_digit') == digit # Checking if the predicted digit is correct
