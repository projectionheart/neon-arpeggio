"""Audio analysis for Neon Arpeggio.

Decodes the song with ffmpeg and computes the per-frame envelopes the
renderer reacts to:

  bass   30-150 Hz energy   -> chroma split on kicks
  hi     4-11 kHz energy    -> city windows flicker with the arpeggio
  onset  spectral flux      -> datamosh tears and block smears

It also estimates the beat grid (BPM + phase) so you can retime the video
to a different song:  python analyze.py my_song.m4a
"""
import subprocess, sys
import numpy as np
import scipy.signal as sg

SR, HOP = 22050, 512


def load_audio(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(SR),
                          '-f', 's16le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def features(path):
    x = load_audio(path)
    f, t, Z = sg.stft(x, SR, nperseg=2048, noverlap=2048 - HOP)
    M = np.abs(Z)
    band = lambda a, b: M[(f >= a) & (f < b)].sum(0)
    flux = np.r_[0, np.maximum(0, np.diff(np.log1p(M * 10), axis=1)).sum(0)]
    onset = np.maximum(flux - sg.medfilt(flux, 31), 0)
    return dict(t=t, bass=band(30, 150), mid=band(150, 2000), hi=band(4000, 11000),
                onset=onset, fps=SR / HOP, duration=len(x) / SR)


def beat_grid(feat, lo=70, hi=180):
    """Autocorrelation tempo guess, then a fine grid search over BPM and phase."""
    o, fps = feat['onset'], feat['fps']
    ac = np.correlate(o, o, 'full')[len(o) - 1:]
    lags = np.arange(1, len(ac)) / fps
    bpm = 60 / lags
    m = (bpm > lo) & (bpm < hi)
    rough = bpm[m][np.argmax(ac[1:][m])]
    best = (-1, rough, 0.0)
    for b in np.arange(rough - 3, rough + 3, 0.02):
        P = 60 / b * fps
        for ph in np.linspace(0, P, 60, endpoint=False):
            idx = np.round(ph + np.arange(0, len(o) - 2 * P, P)).astype(int)
            sc = o[idx].mean()
            if sc > best[0]:
                best = (sc, b, ph / fps)
    return best[1], best[2]


if __name__ == '__main__':
    feat = features(sys.argv[1] if len(sys.argv) > 1 else 'assets/song.m4a')
    bpm, phase = beat_grid(feat)
    print(f'duration {feat["duration"]:.2f}s  tempo {bpm:.2f} BPM  first beat {phase:.3f}s')
