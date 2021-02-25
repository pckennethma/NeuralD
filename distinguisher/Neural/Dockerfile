FROM nvcr.io/nvidia/pytorch:20.12-py3


RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && \
    python3 setup.py install && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
RUN pip install --no-cache-dir dill

COPY ./*.py /workspace/
