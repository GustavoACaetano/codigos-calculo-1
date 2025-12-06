import numpy as np
import matplotlib.pyplot as plt
import wave

# Configuração do áudio sintético
fs = 8000  # taxa de amostragem (8 kHz)
tempo = 0.02 # duração do áudio
t = np.linspace(0, tempo, int(fs * tempo))

# Sinal: seno de 440 Hz + ruído branco
x = 0.7 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.random.randn(len(t))

# Potência instantânea (sinal ao quadrado)
x2 = x ** 2

# --- SALVAR WAV ---
# Normalizar para int16 (formato de áudio)
x_norm = x / np.max(np.abs(x))
x_int16 = (x_norm * 32767).astype(np.int16)

# Criar arquivo WAV
with wave.open("sinal.wav", "wb") as wf:
    wf.setnchannels(1)       # mono
    wf.setsampwidth(2)       # 16 bits
    wf.setframerate(fs)      # 8000 Hz
    wf.writeframes(x_int16.tobytes())

print("Arquivo sinal.wav gerado!")
# --- PLOT ---
plt.figure(figsize=(10, 6))
plt.plot(t, x, label="Sinal real x(t)")
plt.plot(t, x2, label="Potência x(t)²")

plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.title("Sinal de Áudio Real Sintético e sua Potência")
plt.grid(True)
plt.legend()
plt.show()