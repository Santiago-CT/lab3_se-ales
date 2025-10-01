import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

SAMPLE_RATE = 44100  # Frecuencia de muestreo estándar para audio

def record_audio(duration_seconds: int) -> np.ndarray:
    """
    Graba audio desde el micrófono por una duración determinada.
    """
    print(f"Grabando por {duration_seconds} segundos...")
    recording = sd.rec(int(duration_seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  # Espera a que la grabación termine
    print("Grabación finalizada.")
    # Asegurarse de que el audio sea un vector 1D
    return recording.flatten()

def save_wav(filename: str, audio_data: np.ndarray):
    """
    Guarda los datos de audio en un archivo .wav.
    """
    write(filename, SAMPLE_RATE, audio_data)