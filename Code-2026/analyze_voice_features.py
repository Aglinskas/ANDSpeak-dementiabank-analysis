import os
import sys
import subprocess
import pandas as pd
import numpy as np

def analyze_voice_features(input_path, output_path):
    """
    Analyzes voice features from an audio file using DisVoice.
    
    Args:
        input_path (str): Path to the input audio file (mp3 or wav).
        output_path (str): Path to save the extracted features (csv).
    """
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    # Setup paths
    current_dir = os.getcwd()
    # Assuming DisVoice is in the current directory or relative to this script
    # Adjust logic if DisVoice is elsewhere. 
    # Since this script is likely in the project root where DisVoice folder is:
    disvoice_path = os.path.join(current_dir, "DisVoice")
    
    if not os.path.exists(disvoice_path):
        # Fallback: try to find DisVoice relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        disvoice_path = os.path.join(script_dir, "DisVoice")
        
    if not os.path.exists(disvoice_path):
         print("Warning: DisVoice directory not found. Feature extraction may fail.")

    # Convert to WAV if needed
    if input_path.lower().endswith('.mp3'):
        wav_path = input_path.rsplit('.', 1)[0] + ".wav"
        print(f"Converting {input_path} to {wav_path}...")
        try:
            ffmpeg_cmd = './ffmpeg' if os.path.exists('./ffmpeg') else 'ffmpeg'
            subprocess.call([ffmpeg_cmd, '-i', input_path, wav_path, '-y'], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Error converting audio: {e}")
            return
        
        if not os.path.exists(wav_path):
            print("Conversion failed. WAV file not created.")
            return
        
        # Use the converted wav path
        target_audio = wav_path
    else:
        target_audio = input_path

    # Robust Import Strategy for DisVoice
    try:
        # Add submodules to sys.path
        sys.path.append(os.path.join(disvoice_path, "disvoice", "prosody"))
        sys.path.append(os.path.join(disvoice_path, "disvoice", "articulation"))
        sys.path.append(os.path.join(disvoice_path, "disvoice", "glottal"))
        sys.path.append(os.path.join(disvoice_path, "disvoice", "phonation"))
        
        import prosody as prosody_mod
        from articulation import Articulation
        from glottal import Glottal
        from phonation import Phonation
        
        Prosody = prosody_mod.Prosody
        
    except ImportError as e:
        print(f"Failed to import DisVoice modules: {e}")
        return

    features_list = []
    
    # 1. Prosody
    try:
        print("Extracting Prosody features...")
        pro = Prosody()
        feat_pro = pro.extract_features_file(target_audio, static=True, plots=False, fmt="dataframe")
        features_list.append(feat_pro)
    except Exception as e:
        print(f"Prosody extraction failed: {e}")

    # 2. Articulation
    try:
        print("Extracting Articulation features...")
        art = Articulation()
        feat_art = art.extract_features_file(target_audio, static=True, plots=False, fmt="dataframe")
        features_list.append(feat_art)
    except Exception as e:
        print(f"Articulation extraction failed: {e}")

    # 3. Phonation
    try:
        print("Extracting Phonation features...")
        pho = Phonation()
        feat_pho = pho.extract_features_file(target_audio, static=True, plots=False, fmt="dataframe")
        features_list.append(feat_pho)
    except Exception as e:
        print(f"Phonation extraction failed: {e}")

    # 4. Glottal
    try:
        print("Extracting Glottal features...")
        glo = Glottal()
        feat_glo = glo.extract_features_file(target_audio, static=True, plots=False, fmt="dataframe")
        features_list.append(feat_glo)
    except Exception as e:
        print(f"Glottal extraction failed: {e}")

    # Combine and Save
    if features_list:
        all_features = pd.concat(features_list, axis=1)
        all_features.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")
    else:
        print("No features extracted.")

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) == 3:
        input_p = sys.argv[1]
        output_p = sys.argv[2]
        analyze_voice_features(input_p, output_p)
    else:
        # Default for testing
        if os.path.exists("output.mp3"):
            analyze_voice_features("output.mp3", "voice_features.csv")
        else:
            print("Usage: python analyze_voice_features.py <input_audio> <output_csv>")
