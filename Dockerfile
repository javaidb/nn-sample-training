# Use the official PyTorch image with CUDA 12.1 and cuDNN 8 for GPU support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Install JupyterLab, Jupyter Notebook, and IPython kernel
RUN pip install jupyterlab notebook ipykernel

# Install git to clone repositories if needed
RUN apt-get update && apt-get install -y git

# Set the working directory
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt /app/

# Install the dependencies listed in the requirements.txt
RUN pip install -r requirements.txt

# Expose port 8888 for Jupyter Notebook to access
EXPOSE 8888

# Set the container's default command to launch Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
