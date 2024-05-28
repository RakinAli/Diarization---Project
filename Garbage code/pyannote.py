#!/usr/bin/env python
# coding: utf-8

# # Speaker diarization.
# In this project we wish to do Speaker Diarization. Specifically we wish to build custom pipelines in order to answer the following questions:
# - Which embeddings are best for speaker diarization?
# - Which clustering algorithms are best for speaker diarization?
# - Which embedding + Clustering algorithm combination is best for speaker diarization?
# - Does Clustering Algorithms on Deep Neural Network embeddings outperform traditional clustering algorithms?
# - Can end-to-end Deep Neural Network models outperform traditional clustering algorithms?
# - Which Voice activity detections to use for speaker diarization?
# 
# 
# The ground truth are the RTTM files. The RTTM files are in the following format:
# ```
# 
# SPEAKER <NA> 1 0.00 0.39 <NA> <NA> spk_0 <NA>
# SPEAKER <NA> 1 0.39 0.01 <NA> <NA> spk_1 <NA>
# 
# ```
# The first number is the start time, the second number is the duration, and the last number is the speaker id.
# 
# 
# 
#     

# ## Which embeddings are best for speaker diarization?
# ![image.png](attachment:image.png)
# 
# The focus being on the embeddings, we will use the following embeddings:
# - I-vector
# - D-vectors (Deep Speaker Embeddings)

# ### All imports 

# In[54]:


# Typical imports
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import speechbrain as sb

# Scipy
from scipy.spatial.distance import cdist


# All Pyannote imports
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation
from pyannote.audio import Model, Inference
from pyannote.audio import Audio

# Loading the envs
load_dotenv("auths.env")
api_key = os.getenv("API_KEY")


# ### Loading the train data and the ground truth on the train data
# - Below is a playground to load the train data and the ground truth on the train data for one of the files. Later this will be done on all the files. 


train_data_path = "../Dataset/Audio/Dev"
train_label_path = "../Dataset/RTTMs/Dev"

# Experimental data --> just one audio file and its corresponding label
dummy_train_data_path = "../Dataset/Audio/Dev/ahnss.wav"
dummy_train_label_path = "../Dataset/RTTMs/Dev/ahnss.rttm"


# Load the pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection", use_auth_token=api_key
)
pipeline.to(torch.device("cuda"))

# 1. Voice Activity Detection
vad_pipeline = pipeline(dummy_train_data_path)
vad_timeline = vad_pipeline.get_timeline().support()

# 2. Overlapped speech detection
osd_pipeline = Pipeline.from_pretrained(
    "pyannote/overlapped-speech-detection", use_auth_token=api_key
)
output = osd_pipeline(dummy_train_data_path)
osd_timeline = output.get_timeline().support()

# Combine the two timelines 
combined_timeline = vad_timeline.union(osd_timeline)
combined_annotation = Annotation()
for segment in combined_timeline:
    combined_annotation[segment] = "speech"



# In[62]:


# Load the speaker embedding model
embedding_model = Model.from_pretrained(
    "pyannote/embedding", use_auth_token=api_key
)
inference = Inference(embedding_model, window="whole")

# Initialize audio utility
audio = Audio(sample_rate=16000)

# Extract embeddings for each segment
embeddings = []
segments = []

min_length = 160  # Increase the minimum length to a more appropriate size for the model

for segment in tqdm(combined_annotation.itersegments(), desc="Extracting embeddings"):
    waveform, _ = audio.crop(dummy_train_data_path, segment)

    # Ensure the segment is long enough
    if waveform.shape[1] < min_length:
        padding = min_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    embedding = inference({"waveform": waveform, "sample_rate": 16000})
    embeddings.append(embedding)  # Directly append the numpy array
    segments.append(segment)


# %%
