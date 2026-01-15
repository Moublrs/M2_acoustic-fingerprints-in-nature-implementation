import os,io
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

import librosa
from maad import sound
from maad import util
from dotenv import load_dotenv
load_dotenv()
import boto3
s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("WASABI_ENDPOINT"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
)
def load_audio(path):
    """
    Load audio from either:
      - local path: /foo/bar/file.flac
      - S3/Wasabi path: s3://bucket/path/to/file.flac
    Returns: (signal, samplerate)
    """
    if path.startswith("s3://"):
        # Parse s3://bucket/key/like/this.flac
        # "s3:", "", "bucket", "key...", "..."
        parts = path.split("/", 3)
        if len(parts) < 4:
            raise ValueError(f"Invalid S3 path: {path}")
        bucket = parts[2]
        key = parts[3]

        obj = s3.get_object(Bucket=bucket, Key=key)
        audio_bytes = obj["Body"].read()

        # Use librosa to load from bytes
        # sr=None -> keep original sampling rate
        # mono=True (default) -> you can set mono=False if you want stereo
        signal, fs = librosa.load(io.BytesIO(audio_bytes), sr=None)

        return signal, fs

    # Fallback: local file – your original behavior
    return sound.load(path)


def get_image_from_spectrogram(Sxx, extent, db_range=96, gain=0, log_scale=True,
                               interpolation='bilinear', **kwargs):
    """
    Build spectrogram representation and return an image.

    Parameters
    ----------
    Sxx : 2d ndarray
        Spectrogram computed using `maad.sound.spectrogram`.
    extent : list of scalars [left, right, bottom, top]
        Location, in data-coordinates, of the lower-left and upper-right corners.
    db_range : int or float, optional
        Range of values to display the spectrogram. The default is 96.
    gain : int or float, optional
        Gain in decibels added to the signal for display. The default is 0.
    **kwargs : matplotlib image properties
            Other keyword arguments that are passed down to matplotlib.axes.

    Plot the spectrogram of an audio.

    """
    # convert Sxx into dB with values between 0 to 96dB corresponding to a 16 bits soundwave
    Sxx_dB = util.power2dB(Sxx, db_range, gain) + 96

    # OPTIONAL : rescale the values between 0 to 255 (8 bits) depending on the min and max threshold values
    min_threshold = 0
    max_threshold = 50  # value in dB from 0dB (the image will be white) to 96dB (the highest range possible)
    Sxx_dB = Sxx_dB - min_threshold
    Sxx_dB = (Sxx_dB / max_threshold) * 255

    # create an object image
    spectro_image = Image.fromarray(Sxx_dB)
    # rotate the image in orde to have the lowest frequency at the bottom of the image
    spectro_image = spectro_image.transpose(Image.FLIP_TOP_BOTTOM)
    # spectro_image = spectro_image.rotate(180)
    # convert into greyscale (8 bits : 0-255)
    spectro_image = spectro_image.convert("L")

    return spectro_image


def get_spectrograms_from_file(path, window, min_freq, max_freq):
    s, fs = load_audio(path)
    slices = sound.wave2frames(s, int(fs * window))
    slices = np.transpose(slices)

    spectrograms = []
    for slc in slices:
        Sxx, tn, fn, ext = sound.spectrogram(slc, fs,
                                             window='hann', flims=[min_freq, max_freq],
                                             nperseg=1024, noverlap=512)

        spectrograms.append(Sxx)

    spectrograms = np.array(spectrograms)

    # here I am catching an error when the loaded audio file is corrupted
    if 'tn' not in locals() or 'fn' not in locals() or 'ext' not in locals():
        print('XXXXXXXXXXXXX>>>>>>>>>>>>> SOMETHING SEEMS TO BE WRONG WITH THE INPUT FILE HERE: ', path)

    return spectrograms, slices, tn, fn, ext

def get_images_from_spectrograms(spectrograms, extent):
    images = []
    for spectrogram in spectrograms:
        images.append(get_image_from_spectrogram(spectrogram, extent))

    return images


def save_images(path, file_name, images):
    counter=0
    for image in images:
        aux = os.path.join(path, file_name+'_'+str(counter)+'.png')
        image.save(aux)

        counter = counter+1

def build_spectrogram_images_from_audio_file(audiofilepath, outputdirpath, window_t, min_f, max_f):
    """
    Build and save spectrogram images built from audio file
    """

    spectrograms, slices, tn, fn, ext = get_spectrograms_from_file(audiofilepath, window_t, min_f, max_f)
    images = get_images_from_spectrograms(spectrograms, ext)
    #figures = get_figures_from_spectrograms(spectrograms, ext)

    CHECK_FOLDER = os.path.isdir(outputdirpath)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        Path(outputdirpath).mkdir(parents=True, exist_ok=True)

    save_images(outputdirpath, Path(audiofilepath).stem, images)


def navigate_directory_tree(size, rank, input_directory, output_directory, window_t=10, min_f=0, max_f=20000):
    """
    Build and save spectrogram images built from audio files from directory trees
    """
    checkpoint_file_name = os.path.join(output_directory, 'Checkpoint_rank_' + str(rank) + '.txt')
    if not os.path.isfile(checkpoint_file_name):
        # iterate over files in
        # that input_directory
        # and then over subdirectories recursively
        files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
        directories = [d for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d))]
        counter = 0
        for filename in files:
            if counter % size == rank:
                # print('From rank ', rank, ' From ', os.path.join(input_directory, filename))
                # print('From rank ', rank, ' To ', output_directory)


                build_spectrogram_images_from_audio_file(os.path.join(input_directory, filename), output_directory, window_t, min_f, max_f)

            counter = counter + 1

        for dirname in directories:
            navigate_directory_tree(size=size, rank=rank,
                                    input_directory=os.path.join(input_directory, dirname),
                                    output_directory=os.path.join(output_directory, dirname), window_t=window_t, min_f=min_f, max_f=max_f)

        os.makedirs(os.path.dirname(checkpoint_file_name), exist_ok=True)
        with open(checkpoint_file_name, 'w') as f:
            f.write('Files in this directory were completed for rank ' + str(rank))
            f.close()

        print('------>>>>>>> Checkpoint completed for directory: ' + output_directory + ' for rank ' + str(rank))
    else:
        print('------>>>>>>> Checkpoint already completed for directory: ' + output_directory + ' for rank ' + str(rank))