# Speaker-ViT
PyTorch implementation of "Speaker-ViT: global and local vision transformer for speaker verification".

## model class and pretrained checkpoint
we provide the code of model class and pretrained checkpoint, which can be used to verify the paper's results on VoxCeleb1 test set.

There is an example for using this pretrained model to calcute speaker embedding(400 dim).
```python
import torch
import soundfile as sf
from speaker_vit import SpeakerViT
model = SpeakerViT()
model.load_state_dict(torch.load("./speaker-vit.pt"))
model.eval()

wave, _ = sf.read("your_wav_file_path")
tensor_wave = torch.FloatTensor(wave).view(1, -1)
speaker_embedding = model(tensor_wave)
```
||VoxCeleb1-O|VoxCeleb1-E|VoxCeleb1-H|
|:---|:---:|:---:|:---:|
|EER|0.93|1.08|2.05|
|DCF_0.01|0.1047|0.1216|0.2073|
|DCF_0.001|0.2004|0.2212|0.3338|
