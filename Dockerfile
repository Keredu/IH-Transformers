# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:23.03-py3

# Set the working directory
WORKDIR /workspace

# Copy files into the container
COPY requirements.txt /workspace/requirements.txt

RUN pip install -r requirements.txt

COPY scripts/cifar10_train.py /workspace/cifar10_train.py

# Command to run when the container starts
# CMD ["python", "-c", "import transformers; print(transformers.__version__)"]
# CMD ["python", "cifar10_train.py"]
CMD ["/bin/bash"]
