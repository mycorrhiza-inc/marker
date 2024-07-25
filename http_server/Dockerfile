FROM python:3.12.4-alpine3.20


ENV VIRTUAL_ENV=/usr/local
# use UV
# ADD --chmod=755 https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh /install.sh
# RUN /install.sh && rm /install.sh

WORKDIR /app
COPY requirements.txt /app/requirements.txt
# Uncomment for debugging python depenancy issue
# COPY requirements-unlocked.txt /app/requirements.txt

RUN pip install --no-cache -r requirements.txt 
COPY . . 

# expose to docker network
EXPOSE 2718

# Command should be in dockerfile
CMD ["python","litestar_server.py"]
