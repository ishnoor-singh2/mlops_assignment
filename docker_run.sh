docker build -t model_train_1 . --no-cache

# Run the Docker container, mounting a host directory to the container's volume
docker run -v models:/app/models model_train_1
