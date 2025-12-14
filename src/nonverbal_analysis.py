import numpy as np
import librosa
import os
import json

# --- 1. THRESHOLDS FOR QUALITATIVE INTERPRETATION ---

TEMPO_FAST = 150.0  
TEMPO_SLOW = 125.0  
PAUSE_TOO_MUCH_PERCENT = 45.0 
PAUSE_MINIMAL_PERCENT = 35.0  

# --- 2. INTERPRETATION FUNCTIONS ---

def interpret_tempo(bpm):
    if bpm > TEMPO_FAST:
        return "too fast"
    elif bpm >= TEMPO_SLOW:
        return "fast"
    else:
        return "slow"

def interpret_pause_by_percent(pause_percent):
    if pause_percent > PAUSE_TOO_MUCH_PERCENT:
        return "too many pauses"
    elif pause_percent <= PAUSE_MINIMAL_PERCENT:
        return "minimal pauses"
    else:
        return "normal pauses"

# --- 3. CORE ANALYSIS FUNCTION ---

def analyze_audio(file_path, silence_threshold=0.015):
    """
    Analyzes an audio file for non-verbal cues (tempo and pause duration).

    Args:
        file_path (str): The path to the temporary audio file.

    Returns:
        dict: Analysis results formatted as strings with units.
    """
    
    try:
        # Load audio data
        y, sr = librosa.load(file_path, sr=16000)
        total_duration = len(y) / sr

        # 1. Tempo Analysis (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = tempo[0]

        # 2. Pause Analysis (RMS)
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        frame_duration = hop_length / sr 

        silent_frames_count = np.sum(rms < silence_threshold)
        total_silent_time_sec = silent_frames_count * frame_duration
        
        pause_percentage = (total_silent_time_sec / total_duration) * 100 if total_duration > 0 else 0.0
            
        # 3. Qualitative Summary
        tempo_qualitative = interpret_tempo(tempo)
        pause_qualitative = interpret_pause_by_percent(pause_percentage)
        
        summary = f"{tempo_qualitative} tempo and {pause_qualitative}"
            
        # Final Result Dictionary
        result = {
            "tempo_bpm": f"{tempo:.2f} per minute",
            "total_pause_seconds": f"{total_silent_time_sec:.2f} seconds",
            "qualitative_summary": summary,
            "total_duration_seconds": float(f"{total_duration:.2f}") 
        }
        return result

    except Exception as e:
        return {
            "error": f"Failed Non-Verbal Analysis: {str(e)}"
        }

if __name__ == '__main__':
    print("Non-Verbal Analysis Module is ready.")
