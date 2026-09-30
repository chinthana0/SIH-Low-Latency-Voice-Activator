import librosa
import numpy as np

SAMPLE_RATE = 16000
DURATION = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)


def load_audio(file_path):
    """
    Load audio and convert it to a fixed-length 1-second waveform.
    """

    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) < NUM_SAMPLES:
        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio))
        )
    else:
        audio = audio[:NUM_SAMPLES]

    return audio


def extract_mfcc(audio):
    """
    Convert audio waveform into MFCC features.
    """

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=13
    )

    return mfcc
