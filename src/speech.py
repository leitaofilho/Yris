# speech.py
from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile


class Speech:
    def __init__(self, model_name):
        self.model = VitsModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def text_to_speech(self, text, output_filename):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs).waveform
        if isinstance(output, torch.Tensor):
            output = output.squeeze().numpy()
        output = output / max(abs(output))
        output_int16 = (output * 32767).astype("int16")
        scipy.io.wavfile.write(output_filename, rate=self.model.config.sampling_rate, data=output_int16)
