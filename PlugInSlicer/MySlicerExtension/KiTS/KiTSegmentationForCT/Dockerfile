FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
LABEL maintainer="giammarco.labarbera93@gmail.com"

RUN pip install -U --timeout 1000 nibabel
RUN pip install -U pillow==8.3.2
RUN pip install -U --timeout 1000 scikit-image


ADD KiTS_build /kits_build/

WORKDIR /kits_build

# The code to run when container is started:
ENTRYPOINT ["python", "./main_necker.py"]