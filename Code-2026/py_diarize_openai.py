# conda activate andspeak-demo
# cd /Users/aidasaglinskas/Desktop/ANDSpeak/Google_drive_things/Code-2026
# ipython

import pandas as pd
import base64
from openai import OpenAI
import os
from tqdm import tqdm
import numpy as np


def diarized_to_csv(transcription, out_csv=None):
    """
    Convert OpenAI diarized transcription object to CSV.

    Parameters
    ----------
    transcription : TranscriptionDiarized
        Object returned by OpenAI API
    out_csv : str
        Path to output CSV
    """

    rows = []

    for seg in transcription.segments:
        rows.append({
            "segment_id": seg.id,
            "speaker": seg.speaker,
            "start_sec": seg.start,
            "end_sec": seg.end,
            "duration_sec": seg.end - seg.start,
            "text": seg.text.strip()
        })

    df = pd.DataFrame(rows)

    # useful ordering guarantee
    df = df.sort_values("start_sec").reset_index(drop=True)

    if out_csv:
        df.to_csv(out_csv, index=False)

    return df


api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

df = pd.read_csv('../DementiaBank-preprocessed2/file_list.csv')

def get_transcript(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe-diarize",
            file=audio_file,
            response_format="diarized_json",
            chunking_strategy="auto",
            language="en",
            temperature=0)
    return transcript




idx = np.random.permutation(np.arange(len(df)))
for i in tqdm(idx):
    path = df['path'].values[i]
    ofdir = '../DementiaBank-preprocessed2/01-diarized-transcripts-v1/'
    ofn = df['filename'].values[i].replace('.mp3','.csv')
    out_filename = os.path.join(ofdir,ofn)

    if not os.path.exists(out_filename):
        try:
            transcript = get_transcript(path)
            diarized_csv = diarized_to_csv(transcript,out_filename)
        except:
            print(f'errored on {out_filename}')
