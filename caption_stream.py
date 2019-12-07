import builtins
import collections
import os
import os.path
import queue
import sys
import threading
import time

import numpy as np
from scipy import signal

import deepspeech
import obspython as obs
import pyaudio
import webrtcvad

vad_aggressiveness = 2
beam_width = 500
lm_alpha = 0.75
lm_beta = 1.85
source_name = ""
caption_clear_delay = 2.0

audio_thread = None
stop_thread = threading.Event()

__print_prefix = "[DeepSpeech] "


def print(*objs, **kwargs):
    builtins.print(__print_prefix, *objs, **kwargs)


class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, input_rate=RATE_PROCESS):
        self.buffer_queue = queue.Queue()
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            "format": self.FORMAT,
            "channels": self.CHANNELS,
            "rate": self.input_rate,
            "input": True,
            "frames_per_buffer": self.block_size_input,
            "stream_callback": self.stream_callback,
        }

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def stream_callback(self, in_data, frame_count, time_info, status):
        self.buffer_queue.put(in_data)
        return None, pyaudio.paContinue

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(), input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(
        lambda self: 1000 * self.block_size // self.sample_rate
    )


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=2, input_rate=None):
        super().__init__(input_rate=input_rate)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None, stop_thread=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None:
            frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        last_yield = time.time()

        for frame in frames:
            if stop_thread().is_set():
                # Have to yield something so that
                # the for loop in main() breaks
                yield None
                return

            if time.time() - last_yield > caption_clear_delay:
                update_caption("")
                last_yield = time.time()

            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    last_yield = time.time()
                    ring_buffer.clear()


def script_load(settings):
    global audio_thread
    audio_thread = threading.Thread(target=main, args=(lambda: stop_thread,))
    audio_thread.start()


def script_unload():
    global audio_thread
    global stop_thread
    if audio_thread is not None:
        stop_thread.set()
        audio_thread.join()
    print("Unloaded")


def main(stop_thread):
    print("Audio thread started")
    global vad_aggressiveness
    global lm_alpha
    global lm_beta
    global beam_width

    model_path = f"{script_path()}deepspeech-models"
    lm = "lm.binary"
    trie = "trie"
    rate = 16000

    # Load DeepSpeech model
    if os.path.isdir(model_path):
        model_dir = model_path
        model_path = os.path.join(model_dir, "output_graph.pb")
        lm = os.path.join(model_dir, lm)
        trie = os.path.join(model_dir, trie)

    print("Initializing model...")
    print(f"model: {model_path}")
    model = deepspeech.Model(model_path, beam_width)
    if lm and trie:
        print(f"lm: {lm}")
        print(f"trie: {trie}")
        model.enableDecoderWithLM(lm, trie, lm_alpha, lm_beta)

    # Start listening to audio
    vad_audio = VADAudio(aggressiveness=vad_aggressiveness, input_rate=rate)

    frames = vad_audio.vad_collector(stop_thread=stop_thread)

    stream_context = model.createStream()

    print("Listening...")

    for frame in frames:
        if stop_thread().is_set():
            break
        if frame is not None:
            model.feedAudioContent(stream_context, np.frombuffer(frame, np.int16))
        else:
            text = model.finishStream(stream_context)
            print("Recognized: %s" % text)
            update_caption(text)
            stream_context = model.createStream()


def update_caption(text):
    global source_name

    source = obs.obs_get_source_by_name(source_name)
    if source is not None:
        settings = obs.obs_data_create()
        obs.obs_data_set_string(settings, "text", text)
        obs.obs_source_update(source, settings)
        obs.obs_data_release(settings)
        obs.obs_source_release(source)


def script_description():
    return "Realtime captions using DeepSpeech.\nhttps://github.com/mozilla/DeepSpeech\n\nAuthor: Lauri Räsänen"


def script_update(settings):
    global vad_aggressiveness
    global lm_alpha
    global lm_beta
    global beam_width
    global source_name
    global caption_clear_delay

    vad_aggressiveness = obs.obs_data_get_int(settings, "vad_aggressiveness")
    beam_width = obs.obs_data_get_int(settings, "beam_width")
    lm_alpha = obs.obs_data_get_double(settings, "lm_alpha")
    lm_beta = obs.obs_data_get_double(settings, "lm_beta")
    caption_clear_delay = obs.obs_data_get_double(settings, "caption_clear_delay")
    source_name = obs.obs_data_get_string(settings, "source")


def script_defaults(settings):
    obs.obs_data_set_default_int(settings, "vad_aggressiveness", 2)
    obs.obs_data_set_default_int(settings, "beam_width", 500)
    obs.obs_data_set_default_double(settings, "lm_alpha", 0.75)
    obs.obs_data_set_default_double(settings, "lm_beta", 1.85)
    obs.obs_data_set_default_double(settings, "caption_clear_delay", 2.0)


def script_properties():
    props = obs.obs_properties_create()

    obs.obs_properties_add_int(
        props, "vad_aggressiveness", "VAD Aggressiveness", 0, 3, 1
    )
    obs.obs_properties_add_int(props, "beam_width", "Beam Width", 0, 1000, 1)

    obs.obs_properties_add_float(props, "lm_alpha", "LM Alpha", 0, 10.0, 0.1)
    obs.obs_properties_add_float(props, "lm_beta", "LM Beta", 0, 10.0, 0.1)
    obs.obs_properties_add_float(
        props, "caption_clear_delay", "Caption Clear Delay", 0, 100.0, 0.1
    )

    p = obs.obs_properties_add_list(
        props,
        "source",
        "Captions Text Source",
        obs.OBS_COMBO_TYPE_EDITABLE,
        obs.OBS_COMBO_FORMAT_STRING,
    )
    sources = obs.obs_enum_sources()
    if sources is not None:
        for source in sources:
            source_id = obs.obs_source_get_id(source)
            if source_id == "text_gdiplus" or source_id == "text_ft2_source":
                name = obs.obs_source_get_name(source)
                obs.obs_property_list_add_string(p, name, name)

        obs.source_list_release(sources)

    return props
