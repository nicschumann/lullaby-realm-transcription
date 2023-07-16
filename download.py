# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
import torch
from faster_whisper import WhisperModel

def download_model():
    # do a dry run of downloading whisper.
    has_cuda = torch.cuda.is_available()
    device = 'cuda' if has_cuda else 'cpu'
    precision = 'float16' if has_cuda else 'int8'
    model_size = 'large-v2'
    model = WhisperModel(model_size, device=device, precision=precision)

if __name__ == "__main__":
    download_model()