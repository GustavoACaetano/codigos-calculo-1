import numpy as np
import matplotlib.pyplot as plt
import wave
import os

# ============================================================
# 1) Sinal de entrada (16 amostras)
# ============================================================
x = np.array([
    0.1, 0, 0.2, -0.01,
    1, 2, 1, -1,
    0.1, -0.1, 0.05, 0,
    2, -2, 1, -1
], dtype=float)

N = len(x)
block_size = 4
blocks = np.split(x, N // block_size)

# ============================================================
# 2) Cálculo do RMS global → threshold
# ============================================================
mean_square = np.mean(x ** 2)
rms_global = np.sqrt(mean_square)
threshold_energy = mean_square * block_size  # mesmo critério

# ============================================================
# 3) Energia por bloco → decide manter ou zerar
# ============================================================
kept_blocks = []
kept_indices = []

for i, block in enumerate(blocks, start=1):
    energy = np.sum(block ** 2)
    
    if energy >= threshold_energy:
        # mantém o bloco original
        kept_blocks.append(block)
        kept_indices.append(i)
    else:
        # mantém a posição, mas zera o conteúdo
        kept_blocks.append(np.zeros_like(block))

# Reconstrói o sinal filtrado (mesmo tamanho do original)
filtered_signal = np.concatenate(kept_blocks)

# ============================================================
# 4) Repetir áudio para gerar WAV audível
# ============================================================
repeat_factor = 300  # repete 300 vezes → ~0.3s de áudio

x_long = np.tile(x, repeat_factor)
filtered_long = np.tile(filtered_signal, repeat_factor) if filtered_signal.size > 0 else np.zeros(1000)

# ============================================================
# 5) Função para salvar WAV
# ============================================================
def save_wav(path, signal, fs=16000):
    """Salva um array float como WAV PCM16, com normalização automática."""
    if signal.size == 0:
        signal = np.zeros(100, dtype=float)

    max_val = np.max(np.abs(signal))
    if max_val == 0:
        signal_norm = signal
    else:
        signal_norm = signal * (0.9 * 32767 / max_val)

    signal_int16 = signal_norm.astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(signal_int16.tobytes())

# salvar arquivos
original_path = os.path.join("original.wav")
filtered_path = os.path.join("filtered.wav")

save_wav(original_path, x_long)
save_wav(filtered_path, filtered_long)

# ============================================================
# 6) GRÁFICOS
# ============================================================

# --- Sinal original ---
plt.figure(figsize=(8, 2.5))
plt.plot(x)
plt.title("Sinal Original (16 amostras)")
plt.xlabel("Índice")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# --- Sinal filtrado ---
plt.figure(figsize=(8, 2.5))
plt.plot(filtered_signal)
plt.title("Sinal Filtrado (apenas blocos mantidos)")
plt.xlabel("Índice")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# --- Sinal original repetido (para o WAV) ---
plt.figure(figsize=(8, 2.5))
plt.plot(x_long[:200])  # mostra só o início para não ficar gigante
plt.title("Sinal Original Repetido (para áudio)")
plt.xlabel("Índice")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# --- Sinal filtrado repetido (para o WAV) ---
plt.figure(figsize=(8, 2.5))
plt.plot(filtered_long[:200])  # mostra só o início para não ficar gigante
plt.title("Sinal Filtrado Repetido (para áudio)")
plt.xlabel("Índice")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# ============================================================
# 7) Informações finais
# ============================================================
print("RMS global:", rms_global)
print("Threshold de energia:", threshold_energy)
print("Blocos mantidos:", kept_indices)
print("\nArquivos gerados:")
print(" -", original_path)
print(" -", filtered_path)
