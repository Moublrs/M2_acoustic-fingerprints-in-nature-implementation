from dotenv import load_dotenv
load_dotenv()

import os
import boto3
import soundfile as sf
import io
from audio_to_spectrogram import build_spectrogram_images_from_audio_file
# --- Configure client for Wasabi ---

audiofilepath = "s3://ecosurf-dataset/2020/S4A09154/Data/S4A09154_20200103_213000.flac"
build_spectrogram_images_from_audio_file(audiofilepath, '../genereated_spect_images', 10, 0, 16000)


