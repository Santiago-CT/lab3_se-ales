# feature_extractor.py
import numpy as np

def process_signal(audio_data: np.ndarray, num_sub_bands: int = 2) -> tuple:
    """
    Calcula la FFT, el espectro de magnitud, las frecuencias y las energías de las sub-bandas.
    """
    N = len(audio_data)
    if N == 0:
        return None, None, np.zeros(num_sub_bands)

    # Calcular la FFT y el espectro de magnitud (solo la primera mitad)
    fft_result = np.fft.fft(audio_data)[:N // 2]
    fft_magnitude = np.abs(fft_result)
    
    # Calcular el vector de frecuencias para el eje X del gráfico
    freqs = np.fft.fftfreq(N, 1 / 44100)[:N // 2]

    # Dividir el espectro y calcular la energía de cada sub-banda
    sub_band_size = len(fft_magnitude) // num_sub_bands
    energies = []
    for i in range(num_sub_bands):
        start = i * sub_band_size
        end = start + sub_band_size
        sub_band = fft_magnitude[start:end]
        
        # Fórmula de energía: (1/N) * sum(|X(k)|^2)
        energy = (1 / N) * np.sum(sub_band**2)
        energies.append(energy)

    return freqs, fft_magnitude, np.array(energies)