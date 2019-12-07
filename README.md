# deepspeech\_obs\_captions
Realtime captions in [obs-studio](https://github.com/obsproject/obs-studio) using [DeepSpeech](https://github.com/mozilla/DeepSpeech).

## Installation
- Install Python 3.6 64-bit, or 32-bit depending on your OBS version.
- `pip install -r ./requirements.txt`
- Download trained models from [https://github.com/mozilla/DeepSpeech/releases](https://github.com/mozilla/DeepSpeech/releases) and extract them into `./deepspeech-models`.

**The folder structure should look something like:**  
- `caption_stream.py`
- `deepspeech-models`
  - `lm.binary`
  - `output_graph.pb`
  - `output_graph.pbmm`
  - `output_graph.tflite`
  - `trie`

## OBS Setup
- Add a Text source
- Navigate to `Tools > Scripts`
- Set your Python path in `Python Settings` tab
- Add `./caption_stream.py` in `Scripts` tab
- Select the script and assign `Captions Text Source`

## Config
`VAD Aggressiveness` controls voice activation, a lower value will let more noise through.  
`Caption Clear Delay` controls how long captions remain on the screen.  
`Captions Text Source` controls what element captions appear on.  
  
The other settings are probably best left alone unless you know what you are doing.  
For more details, see: [https://github.com/mozilla/DeepSpeech/blob/master/examples/mic\_vad\_streaming/README.rst#usage](https://github.com/mozilla/DeepSpeech/blob/master/examples/mic_vad_streaming/README.rst#usage)
