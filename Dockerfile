FROM --platform=linux/amd64 pytorch/pytorch
# Use a 'large' base container to show-case how to load tensorflow or pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER root

# install the cv2 dependencies that are normally present on the local machine, but might be missing in your Docker container causing the issue.
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
USER user
WORKDIR /opt/app
COPY --chown=user:user setup.py /opt/app/
COPY --chown=user:user requirements.txt /opt/app/
# You can add any Python dependencies to requirements.txt
#RUN conda env create -f /opt/app/environmentclean.yaml
RUN python3 -m pip install --user --no-cache-dir --no-color --requirement /opt/app/requirements.txt
RUN python3 -m pip install opencv-python-headless

#Copy your python scripts
COPY --chown=user:user scripts/img_gen/inference.py /opt/app/inference.py
COPY --chown=user:user ldm/ /opt/app/ldm/
COPY --chown=user:user configs/ /opt/app/configs/
# TODO: Download the checkpoints first before build
COPY --chown=user:user checkpoints/ /opt/app/checkpoints/
ENTRYPOINT ["python3", "inference.py"]
