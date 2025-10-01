# gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Importar nuestros módulos de lógica
import audio_handler
import feature_extractor

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Laboratorio 2: Reconocimiento de Voz con FFT")
        self.geometry("800x650")

        # Almacenes de datos
        self.recordings_word1 = []
        self.recordings_word2 = []
        self.thresholds = {} # Guardará los vectores de energía promedio

        # --- Interfaz ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Paneles
        train_frame = self._create_train_frame(main_frame)
        train_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        plot_frame = self._create_plot_frame(main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def _create_train_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Fase de Entrenamiento y Reconocimiento")
        
        # Duración de grabación
        ttk.Label(frame, text="Duración (s):").grid(row=0, column=0, padx=5, pady=5)
        self.duration_var = tk.StringVar(value="2")
        ttk.Entry(frame, textvariable=self.duration_var, width=5).grid(row=0, column=1)
        
        # --- Palabra 1 ---
        ttk.Separator(frame, orient='horizontal').grid(row=1, columnspan=3, sticky='ew', pady=10)
        ttk.Label(frame, text="Palabra 1:", font=("Helvetica", 12, "bold")).grid(row=2, column=0, sticky='w')
        ttk.Button(frame, text="Grabar Muestra", command=lambda: self.record_sample(1)).grid(row=3, column=0, padx=5, pady=5)
        self.count_word1 = tk.StringVar(value="Muestras: 0")
        ttk.Label(frame, textvariable=self.count_word1).grid(row=3, column=1)

        # --- Palabra 2 ---
        ttk.Separator(frame, orient='horizontal').grid(row=4, columnspan=3, sticky='ew', pady=10)
        ttk.Label(frame, text="Palabra 2:", font=("Helvetica", 12, "bold")).grid(row=5, column=0, sticky='w')
        ttk.Button(frame, text="Grabar Muestra", command=lambda: self.record_sample(2)).grid(row=6, column=0, padx=5, pady=5)
        self.count_word2 = tk.StringVar(value="Muestras: 0")
        ttk.Label(frame, textvariable=self.count_word2).grid(row=6, column=1)

        # --- Entrenamiento ---
        ttk.Separator(frame, orient='horizontal').grid(row=7, columnspan=3, sticky='ew', pady=10)
        ttk.Button(frame, text="ENTRENAR (Calcular Umbrales)", command=self.train_system).grid(row=8, columnspan=3, pady=10)
        self.thresholds_label = ttk.Label(frame, text="Sistema no entrenado.", wraplength=200)
        self.thresholds_label.grid(row=9, columnspan=3)

        # --- Reconocimiento ---
        ttk.Separator(frame, orient='horizontal').grid(row=10, columnspan=3, sticky='ew', pady=20)
        ttk.Button(frame, text="GRABAR Y RECONOCER", command=self.recognize).grid(row=11, columnspan=3, pady=10, ipady=10)
        self.result_label = ttk.Label(frame, text="Resultado: --", font=("Helvetica", 14, "bold"))
        self.result_label.grid(row=12, columnspan=3, pady=10)

        return frame

    def _create_plot_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Visualización")
        
        self.fig, (self.ax_spec, self.ax_energy) = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1]})
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        return frame

    def record_sample(self, word_id):
        try:
            duration = int(self.duration_var.get())
            audio_data = audio_handler.record_audio(duration)
            
            if word_id == 1:
                self.recordings_word1.append(audio_data)
                self.count_word1.set(f"Muestras: {len(self.recordings_word1)}")
            else:
                self.recordings_word2.append(audio_data)
                self.count_word2.set(f"Muestras: {len(self.recordings_word2)}")
            
            self.update_plots(audio_data)
        except ValueError:
            messagebox.showerror("Error", "La duración debe ser un número entero.")

    def train_system(self):
        if len(self.recordings_word1) < 2 or len(self.recordings_word2) < 2:
            messagebox.showwarning("Faltan Muestras", "Debes grabar al menos 2 muestras de cada palabra para entrenar.")
            return

        energies1 = [feature_extractor.process_signal(rec)[2] for rec in self.recordings_word1]
        avg_energy1 = np.mean(energies1, axis=0)
        
        energies2 = [feature_extractor.process_signal(rec)[2] for rec in self.recordings_word2]
        avg_energy2 = np.mean(energies2, axis=0)

        self.thresholds = {"palabra1": avg_energy1, "palabra2": avg_energy2}
        
        info_text = f"Entrenamiento completado:\n\n"
        info_text += f"Energía Promedio Palabra 1:\nSub-b1: {avg_energy1[0]:.4g} | Sub-b2: {avg_energy1[1]:.4g}\n\n"
        info_text += f"Energía Promedio Palabra 2:\nSub-b1: {avg_energy2[0]:.4g} | Sub-b2: {avg_energy2[1]:.4g}"
        self.thresholds_label.config(text=info_text, foreground="green")

    def recognize(self):
        if not self.thresholds:
            messagebox.showerror("Error", "El sistema no ha sido entrenado.")
            return

        try:
            duration = int(self.duration_var.get())
            unknown_audio = audio_handler.record_audio(duration)
            
            freqs, mag, energies = feature_extractor.process_signal(unknown_audio)
            self.update_plots(unknown_audio, energies)
            
            dist1 = np.linalg.norm(energies - self.thresholds["palabra1"])
            dist2 = np.linalg.norm(energies - self.thresholds["palabra2"])
            
            if dist1 < dist2:
                result, color = "Palabra 1", "blue"
            else:
                result, color = "Palabra 2", "orange"
            
            self.result_label.config(text=f"Resultado: {result}", foreground=color)

        except ValueError:
            messagebox.showerror("Error", "La duración debe ser un número entero.")

    def update_plots(self, audio_data, energies=None):
        self.ax_spec.clear()
        self.ax_energy.clear()
        
        freqs, mag, calc_energies = feature_extractor.process_signal(audio_data)
        
        self.ax_spec.plot(freqs, mag)
        self.ax_spec.set_title("Espectro de Frecuencia (FFT)")
        self.ax_spec.set_xlabel("Frecuencia (Hz)")
        self.ax_spec.set_ylabel("Magnitud")
        
        energies_to_plot = energies if energies is not None else calc_energies
        labels = [f"Sub-banda {i+1}" for i in range(len(energies_to_plot))]
        self.ax_energy.bar(labels, energies_to_plot, color=['skyblue', 'salmon'])
        self.ax_energy.set_title("Energía por Sub-banda")
        self.ax_energy.set_ylabel("Energía")
        
        self.canvas.draw()