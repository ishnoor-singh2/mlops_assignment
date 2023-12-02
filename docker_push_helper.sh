#!/bin/bash

#cmd for building dependency image
docker build -f DependencyDockerfile -t dependencyimage .
docker tag dependencyimage m22aie233.azurecr.io/dependencyimage:latest

# cmd for pushing the Dependency Image to ACR
docker push m22aie233.azurecr.io/dependencyimage:latest

# cmd for building the final image
docker build -f FinalDockerfile -t finalimage .
docker tag finalimage m22aie233.azurecr.io/finalimage:latest

# cmd for pushing the final Image to ACR
docker push m22aie233.azurecr.io/finalimage:latest
