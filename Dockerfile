# Use a base image with Python 3.12
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the project files into the Docker image
COPY . .

# Install the project dependencies
RUN pip install --upgrade pip
RUN pip install -e .

# Set the entry point to run the project
ENTRYPOINT ["exo"]
