# I would have needed to add the code to system path to import it. Just use a Python package instead.

"""
This code generates RTTM files for all the audio files in the test folder with all the models in the custom pipeline

"""
# Custom package
from Simple_diarizer.utils import combined_waveplot, waveplot
from Simple_diarizer.diarizer import Diarizer
from Simple_diarizer.utils import check_ffmpeg

# Standard library
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from tqdm import tqdm
import sys

# Global variables
AUDIO_FILE = "../Dataset/Audio/Test/aggyz.wav"
AUDIO_TEST_PATH = "../Dataset/Audio/Test/"
AUDIO_OUT_PATH = "../Results/Custom pipeline/"
IMAGE_OUT_PATH = "../Images/Audio_vis/"
RTTM_OUT_PATH = "../Results/Custom pipeline/"

def test_diarizer():
    # Testing the diarizer
    custom_diarizer = Diarizer(
        embed_model="xvec", # Or escapa
        cluster_method="sc" # SC- spectral clustering, ahc - agglomerative hierarchical clustering
    )

    segments = custom_diarizer.diarize(AUDIO_FILE, outfile="output")
    signal,fs = sf.read(AUDIO_FILE)
    combined_waveplot(signal, fs, segments, output_path=IMAGE_OUT_PATH)
    print("Working")


def run_all_untrained():
    # Create Diarizer instances
    diarizers = {
        "xvec_sc": Diarizer(embed_model="xvec", cluster_method="sc"),
        "xvec_ahc": Diarizer(embed_model="xvec", cluster_method="ahc"),
        "ecapa_sc": Diarizer(embed_model="ecapa", cluster_method="sc"),
        "ecapa_ahc": Diarizer(embed_model="ecapa", cluster_method="ahc"),
    }

    # Test all models
    for name, diarizer in diarizers.items():
        for audio_file in tqdm(os.listdir(AUDIO_TEST_PATH)):
            if audio_file.endswith(".wav"):
                # Create all the paths
                audio_file_name = audio_file.split(".")[0]
                rttm_out = os.path.join(RTTM_OUT_PATH, name, audio_file_name + ".rttm")
                audio_path = os.path.join(AUDIO_TEST_PATH, audio_file)

                # Check if the RTTM file already exists
                if os.path.exists(rttm_out):
                    print(f"Skipping {audio_file} as it has already been processed.")
                    continue

                # Diarize the audio file from the path and store it in the RTTM path
                segments = diarizer.diarize(audio_path, outfile=rttm_out)
                signal, fs = sf.read(audio_path)

                # Save the image to the path
                output_image_path = os.path.join(IMAGE_OUT_PATH, name, audio_file_name)

                # Do a try-catch to avoid errors --
                try:
                    combined_waveplot(
                        signal, fs, segments, output_path=output_image_path
                    )
                except Exception as e:
                    print(f"Error in file {audio_file}: {e}")
                    continue


"""
Scripts to run some tests
"""
def main():
    # To test the diarizer
    #test_diarizer()

    # To run all untrained models
    run_all_untrained()


if __name__ == "__main__":
    main()
