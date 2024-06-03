# I would have needed to add the code to system path to import it. Just use a Python package instead.

# Custom package
from Simple_diarizer.utils import combined_waveplot, waveplot
from Simple_diarizer.diarizer import Diarizer
from Simple_diarizer.utils import check_ffmpeg

# Standard library
import soundfile as sf
import matplotlib.pyplot as plt
import torch


# Global variables
AUDIO_FILE = "../Dataset/Audio/Test/aggyz.wav"


def test_diarizer():
    # Testing the diarizer
    custom_diarizer = Diarizer(
        embed_model="xvec", # Or escapa
        cluster_method="sc" # SC- spectral clustering, ahc - agglomerative hierarchical clustering
    )

    segments = custom_diarizer.diarize(AUDIO_FILE, outfile="output")
    signal,fs = sf.read(AUDIO_FILE)
    combined_waveplot(signal, fs, segments)
    print("Working")
    plt.show()


"""
Scripts to run some tests
"""
def main():
    
    test_diarizer()


if __name__ == "__main__":
    main()
