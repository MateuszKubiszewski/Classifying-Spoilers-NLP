FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
RUN pip3 install pandas transformers numpy
COPY bert-task-1.py /bert-task-1.py
COPY model /model
COPY tokenizer /tokenizer
ENTRYPOINT [ "" ]