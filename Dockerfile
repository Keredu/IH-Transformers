# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:23.03-py3

# Set the working directory
WORKDIR /workspace

# Copy files into the container
COPY requirements.txt /workspace/requirements.txt

RUN pip install -r requirements.txt

# Command to run when the container starts
CMD ["python", "-c", "import transformers; print(transformers.__version__)"]

