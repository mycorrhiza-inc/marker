# Use an official Python runtime as a parent image
FROM python:3.11.9

# Set the working directory in the container
WORKDIR /app

# Clone the repository



# Install Tesseract
RUN apt-get update && \
  apt-get install -y lsb-release apt-transport-https wget && \
  wget -qO - https://notesalexp.org/debian/alexp_key.asc | apt-key add - && \
  echo "deb https://notesalexp.org/tesseract-ocr5/$(lsb_release -cs)/ $(lsb_release -cs) main" \
  | tee /etc/apt/sources.list.d/notesalexp.list > /dev/null && \
  apt-get update -oAcquire::AllowInsecureRepositories=true && \
  apt-get install -y notesalexp-keyring --allow-unauthenticated && \
  apt-get update && \
  apt-get install -y tesseract-ocr && \
  rm -rf /var/lib/apt/lists/*

# Install Ghostscript
RUN wget https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs10012/ghostscript-10.01.2.tar.gz && \
  tar -xvf ghostscript-10.01.2.tar.gz && \
  cd ghostscript-10.01.2 && \
  ./configure && \
  make install && \
  cd .. && \
  rm -rf ghostscript-10.01.2 ghostscript-10.01.2.tar.gz

# Find the tessdata directory and create a local.env file with the TESSDATA_PREFIX
RUN tessdata_path=$(find / -name tessdata -print -quit) && \
  echo "TESSDATA_PREFIX=${tessdata_path}" > local.env

COPY ./scripts/ .

# Install system requirements
# Note: Scripted installation of tesseract and ghostscript may need adjustments
RUN apt-get update && \
  apt-get install -y $(cat scripts/install/apt-requirements.txt)


# Install system dependencies
RUN apt-get update && apt-get install -y \
  libgl1-mesa-glx \
  libglib2.0-0


# Update PyTorch (adjust this part based on GPU/CPU requirements)
RUN pip install --upgrade pip
COPY ./requirements.txt .

RUN pip install --no-cache -r requirements.txt 
# ### GPU Only
# RUN pip install torch 
# 
# 
# COPY ./pyproject.toml . 
# COPY ./poetry.lock .
# # Install Python dependencies
# RUN pip install poetry
# RUN poetry install

## ====
### CPU Only
# RUN poetry remove torch
# RUN pip install --no-cache-dir --no-warn-script-location torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
## ====
# RUN pip install --no-cache-dir --no-warn-script-location scikit-learn Pillow pydantic pydantic-settings transformers numpy python-dotenv \
#   torch ray tqdm tabulate ftfy texify rapidfuzz surya-ocr \
#  filetype regex pdftext grpcio 

RUN pip install opencv-python-headless

RUN pip install --no-cache-dir --no-warn-script-location  pika


COPY . .

# The command to run the application
# CMD ["poetry", "run", "server.py" , "-port", "2718"]
