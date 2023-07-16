import base64
from potassium import Potassium, Request, Response
from time import perf_counter

from faster_whisper import WhisperModel
import numpy as np
import torch

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    has_cuda = torch.cuda.is_available()
    device = 'cuda' if has_cuda else 'cpu'
    precision = 'float16' if has_cuda else 'int8'
    model_size = 'large-v2'
    model = WhisperModel(model_size, device=device, compute_type=precision)
   
    context = {
        "model": model
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:

    # Get the model
    model : WhisperModel = context.get("model")
    
    # Set start timestamp
    s = perf_counter()
        
    # Decode the request
    string_data = request.json.get("bytes")
    shape = request.json.get("shape")
    beam_size = int(request.json.get("beam_size"))
    language = request.json.get("language")
    encoding = request.json.get('encoding')

    byte_data = string_data.encode(encoding)
    array_data = base64.b64decode(byte_data)
    audio_buffer = np.frombuffer(array_data, dtype=np.float32)

    # Begin transcription    
    segments, _ = model.transcribe(
        audio_buffer,
        beam_size=beam_size,
        language=language,
        vad_filter=False,
        word_timestamps=True
    )

    result = []

    for speech in segments:

        words = list(map(lambda w: {"start": w.start, "end": w.end, "word": w.word, "prob": w.probability}, speech.words))

        segment_result = {
            "speaker": 'User',
            "start": speech.start,
            "end": speech.end,
            "text": speech.text,
            "words": words
        }

        # merge results, if needed...
        if len(result) > 0: # and abs(speech.start - result[-1].end) < 0.05:
            d = speech.words[0].start - result[-1]['words'][-1]['end']
            
            if d < 0.5: # arbitrary threshold
                result[-1]['end'] = speech.end
                result[-1]['text'] += speech.text
                result[-1]['words'] += words

            else:
                result.append(segment_result)

        else:
            result.append(segment_result)

    e = perf_counter()

    response_json = {
        'duration': (e - s),
        'segments': result
    }

    return Response(
        json = response_json, 
        status=200
    )


if __name__ == "__main__":
    app.serve()