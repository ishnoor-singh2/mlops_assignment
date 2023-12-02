#!/bin/bash

# Build the Dependency Image
docker build -f DependencyDockerfile -t dependencyimage .
docker tag dependencyimage m22aie233.azurecr.io/dependencyimage:latest

# Push the Dependency Image to Azure Container Registry
docker push m22aie233.azurecr.io/dependencyimage:latest

# Build the Final Image
docker build -f FinalDockerfile -t finalimage .
docker tag finalimage m22aie233.azurecr.io/finalimage:latest

# Push the Final Image to Azure Container Registry
docker push m22aie233.azurecr.io/finalimage:latest
