FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /workspace

COPY train.py /workspace/train.py
COPY data.txt /workspace/data.txt

CMD ["python", "train.py", "--input", "data.txt", "--epochs", "5"]
