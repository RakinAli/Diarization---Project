import os 
import sys
import numpy as np
import pandas as pd
import tqdm 
import librosa



# Align the MFCC data with the RTM data -- Speaker segmentation for training
def align_mfcc(mfcc_data, sr, rttm_path, hop_length=220):
    """
    Aligns MFCC data with RTTM data using a specified hop length.

    Args:
        mfcc_data (numpy.ndarray): MFCC data.
        sr (int): Sampling rate.
        rttm_path (str): Path to the RTTM file.
        hop_length (int, optional): Number of samples between successive frames. Defaults to 220.

    Returns:
        list of dicts: Each dictionary contains 'Speaker Name' and 'MFCC Segment'.

    Raises:
        KeyError: If required columns are missing in the DataFrame.
    """
    rttm_data = parse_rttm(rttm_path)
    rttm_data = prepare_data(rttm_data)

    if "End Time" not in rttm_data.columns or "Turn Onset" not in rttm_data.columns:
        raise KeyError("Necessary columns are missing from RTTM data.")

    # Convert RTTM times to frame indices
    rttm_data["Start Frame"] = (rttm_data["Turn Onset"] * sr / hop_length).astype(int)
    rttm_data["End Frame"] = (rttm_data["End Time"] * sr / hop_length).astype(int)

    segments = []
    num_frames = mfcc_data.shape[1]
    audio_duration = num_frames * hop_length / sr

    print(f"Audio duration (seconds): {audio_duration}")
    print(f"Number of MFCC frames: {num_frames}")
    print(f"Sampling rate: {sr}")
    print(f"Hop length (samples): {hop_length}")

    for _, row in rttm_data.iterrows():
        start_frame = row["Start Frame"]
        end_frame = row["End Frame"]


        # Clip start_frame and end_frame to valid range
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(0, min(end_frame, num_frames))

        if start_frame >= end_frame:
            print(
                f"Skipping segment with invalid frame range: Start Frame = {start_frame}, End Frame = {end_frame}"
            )
            sys.exit(1)
            continue

        segment_mfcc = mfcc_data[:, start_frame:end_frame]
        segments.append(
            {"Speaker Name": row["Speaker Name"], "MFCC Segment": segment_mfcc}
        )

    return segments


def verify_rttm_audio_length(dev_audio_path, dev_RTM_path):
    files = os.listdir(dev_audio_path)
    issues_found = False
    for file in tqdm.tqdm(files):
        if file.endswith(".wav"):
            file_path = os.path.join(dev_audio_path, file)
            rttm_path = os.path.join(dev_RTM_path, file.replace(".wav", ".rttm"))
            y, sr = librosa.load(file_path)
            audio_duration = librosa.get_duration(y=y, sr=sr)

            rttm_data = parse_rttm(rttm_path)
            rttm_data = prepare_data(rttm_data)
            max_end_time = rttm_data["End Time"].max()

            if max_end_time > audio_duration:
                print(
                    f"File {file} has end time {max_end_time} exceeding audio duration {audio_duration}"
                )
                issues_found = True
    if not issues_found:
        print("No issues found with RTTM annotations exceeding audio duration.")


# Parse the dummy RTTM file
def parse_rttm(file_path):
    columns = [
        "Type",
        "File ID",
        "Channel ID",
        "Turn Onset",
        "Turn Duration",
        "Orthography Field",
        "Speaker Type",
        "Speaker Name",
        "Confidence Score",
        "Signal Lookahead Time",
    ]
    df = pd.read_csv(file_path, sep="\s+", names=columns)
    return df[["Turn Onset", "Turn Duration", "Speaker Name"]]


# Grabs the important data from the RTM file and creats End Time column
def prepare_data(rttm_data):
    rttm_data["Turn Onset"] = rttm_data["Turn Onset"].astype(float)
    rttm_data["Turn Duration"] = rttm_data["Turn Duration"].astype(float)
    rttm_data["End Time"] = rttm_data["Turn Onset"] + rttm_data["Turn Duration"]
    return rttm_data


def calculate_rows(path_to_Rtm):
    # Goes to ""../Dataset/RTMS/Dev" and runes parse_rttm and prepare_data on all files. Concat all the dataframes and return the number of rows
    files = os.listdir(path_to_Rtm)
    data = pd.DataFrame()
    for file in files:
        rttm_data = parse_rttm(os.path.join(path_to_Rtm, file))
        rttm_data = prepare_data(rttm_data)
        data = pd.concat([data, rttm_data])
    return data.shape[0], data  


# Function to gather and align all data
def get_training_and_validation(dev_audio_path, dev_RTM_path):
    files = os.listdir(dev_audio_path)
    all_data = []

    for file in tqdm.tqdm(files):
        if file.endswith(".wav"):
            file_path = os.path.join(dev_audio_path, file)
            rttm_path = os.path.join(dev_RTM_path, file.replace(".wav", ".rttm"))
            mfcc_data, sr = get_mfcc(file_path)
            alligned_data = align_mfcc(mfcc_data, sr, rttm_path)
            all_data.extend(alligned_data)
    return all_data


# Example usage
# verify_rttm_audio_length("../Dataset/Audio/Dev", "../Dataset/RTMS/Dev")
rows,data  = calculate_rows("../Dataset/RTMS/Dev")
print("Total number of rows" , rows)
print("Testing get_training_and_validation")
all_data = get_training_and_validation("../Dataset/Audio/Dev", "../Dataset/RTMS/Dev")

print("Total number of rows" , len(all_data))
