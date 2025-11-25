"""
Dual-TTS helper:
- Nhận text input
- Tách các cụm từ tiếng Anh
- Gọi VieNeu-TTS để tổng hợp phần tiếng Việt
- Gọi viXTTS (Coqui TTS via `TTS` lib) để tổng hợp phần tiếng Anh
- Ghép wave (numpy) và trả về numpy array + samplerate
"""

import re
import numpy as np
import soundfile as sf
import tempfile
import os
from typing import Tuple

# Try to import TTS (viXTTS). If không có, chúng ta mock để dev.
try:
    from TTS.api import TTS as CoquiTTS
    HAVE_COQUI = True
except Exception:
    HAVE_COQUI = False

# Regex đơn giản để xác định từ/chuỗi tiếng Anh (latin letters)
ENG_TOKEN_RE = re.compile(r"[A-Za-z0-9\-\']+")

def split_text_segments(text: str):
    """
    Trả về list của (lang, token) theo thứ tự xuất hiện,
    lang: 'vi' hoặc 'en'
    Simple: group contiguous english tokens as en, others as vi.
    """
    tokens = text.split()
    segments = []
    cur_lang = None
    cur_words = []
    for w in tokens:
        if ENG_TOKEN_RE.fullmatch(w):
            lang = "en"
        else:
            lang = "vi"
        if cur_lang is None:
            cur_lang = lang
            cur_words = [w]
        elif lang == cur_lang:
            cur_words.append(w)
        else:
            segments.append((cur_lang, " ".join(cur_words)))
            cur_lang = lang
            cur_words = [w]
    if cur_lang is not None:
        segments.append((cur_lang, " ".join(cur_words)))
    return segments

def concat_audio_segments(segments_wavs, target_sr=24000, gap_s=0.05):
    """
    segments_wavs: list of numpy arrays (mono)
    target_sr: sample rate
    gap_s: silence gap between segments (seconds)
    Return: numpy float32 array
    """
    gap = np.zeros(int(gap_s * target_sr), dtype=np.float32)
    out = []
    for i, w in enumerate(segments_wavs):
        # ensure float32
        arr = w.astype(np.float32)
        # if stereo, convert to mono
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        out.append(arr)
        if i != len(segments_wavs) - 1:
            out.append(gap)
    if not out:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(out)

class DualTTS:
    def __init__(self, vieneu_tts, vixtts_model_name: str = "tts_models/en/vctk/vits"):
        """
        vieneu_tts: instance of your VieNeuTTS (or compatible) with methods:
           - encode_reference(path) -> ref_codes
           - infer(text, ref_codes, ref_text_raw) -> numpy waveform (float32) at 24000
        vixtts_model_name: coqui TTS model name to use for English (can change)
        """
        self.vieneu = vieneu_tts
        self.target_sr = 24000
        self.vixtts = None
        if HAVE_COQUI:
            try:
                # Tải model viXTTS / Coqui TTS (English)
                self.vixtts = CoquiTTS(model_name=vixtts_model_name)
            except Exception as e:
                print("⚠️ Không tải được Coqui TTS model:", e)
                self.vixtts = None
        else:
            print("⚠️ Coqui TTS (TTS lib) không cài đặt. English TTS sẽ bị mock.")

    def synthesize_segment_vn(self, text_vn: str, ref_audio_path: str, ref_text_raw: str):
        """
        Gọi VieNeu TTS để tổng hợp phần tiếng Việt - trả về numpy waveform float32
        """
        ref_codes = None
        try:
            ref_codes = self.vieneu.encode_reference(ref_audio_path)
        except Exception as e:
            print("⚠️ Lỗi khi encode_reference:", e)
        wav = self.vieneu.infer(text_vn, ref_codes, ref_text_raw)
        # đảm bảo float32 numpy
        arr = np.array(wav, dtype=np.float32)
        return arr

    def synthesize_segment_en(self, text_en: str):
        """
        Gọi viXTTS/Coqui TTS để tổng hợp English; nếu không có Coqui TTS,
        sẽ trả về mảng âm rỗng.
        """
        if not text_en or text_en.strip() == "":
            return np.zeros(0, dtype=np.float32)
        if self.vixtts is None:
            # Mock: phát ra silence trong 0.5s x số từ (fallback development)
            n_words = len(text_en.split())
            dur = max(0.25, 0.12 * n_words)
            print(f"⚠️ ViXTTS không sẵn sàng — trả về silence {dur}s cho {[text_en]}")
            return np.zeros(int(dur * self.target_sr), dtype=np.float32)
        # Coqui TTS trả về wav numpy & sr (tuỳ model); TTS.api.TTS.tts_to_file hoặc tts_to_numpy
        try:
            # tts_to_file tương thích nhưng để lấy numpy dùng tts.tts
            wav = self.vixtts.tts(text_en)
            # Coqui TTS tts() có thể trả về numpy array hoặc filepath; handle both
            if isinstance(wav, str) and os.path.exists(wav):
                arr, sr = sf.read(wav)
                if sr != self.target_sr:
                    # resample if needed (try simple np.repeat/decimate if integer ratio)
                    import math
                    ratio = self.target_sr / sr
                    arr = np.interp(
                        np.arange(0, len(arr) * ratio) / ratio,
                        np.arange(0, len(arr)),
                        arr
                    ).astype(np.float32)
                return arr.astype(np.float32)
            elif isinstance(wav, np.ndarray):
                return wav.astype(np.float32)
            else:
                return np.array(wav, dtype=np.float32)
        except Exception as e:
            print("⚠️ Lỗi khi synthesize EN with viXTTS:", e)
            return np.zeros(0, dtype=np.float32)

    def synthesize_dual(self, full_text: str, ref_audio_path: str, ref_text_raw: str) -> Tuple[np.ndarray, int]:
        """
        Main: tách chuỗi, synth mỗi đoạn phù hợp, ghép lại.
        Trả về (wav_array (float32), samplerate)
        """
        segments = split_text_segments(full_text)
        wav_segments = []
        for lang, seg_text in segments:
            if lang == "vi":
                if seg_text.strip():
                    print("🔊 Synth VN segment:", seg_text)
                    wav_vn = self.synthesize_segment_vn(seg_text, ref_audio_path, ref_text_raw)
                    wav_segments.append(wav_vn)
            else:  # en
                if seg_text.strip():
                    print("🔊 Synth EN segment:", seg_text)
                    wav_en = self.synthesize_segment_en(seg_text)
                    wav_segments.append(wav_en)
        out = concat_audio_segments(wav_segments, target_sr=self.target_sr, gap_s=0.06)
        return out, self.target_sr

# Convenience factory
def make_dual_tts(vieneu_tts, vixtts_model_name="tts_models/en/vctk/vits"):
    return DualTTS(vieneu_tts, vixtts_model_name=vixtts_model_name)
