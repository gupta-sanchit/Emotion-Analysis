import warnings
warnings.simplefilter(action='ignore')


import pandas as pd
import librosa
from glob import glob
from tqdm import tqdm


df = pd.DataFrame(columns=['ZeroCrossingRate','RMSE','Chroma','Centroid','Bandwidth','RollOff','Emotion'],index=range(0,len(audioFiles)))
mfcc_df = pd.DataFrame(columns = [ 'M-' + str(i) for i in range(0,21)],index=range(0,len(audioFiles)))

idx = 0
for file in tqdm(audioFiles):
    
    emotion = file.split("_")[2]
    y, sr = librosa.load(file)
    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y,sr=sr)
    
    for i,e in enumerate(mfcc):
        mfcc_df[f"M-{i}"][idx] = mean(e)
        
    df['ZeroCrossingRate'][idx] = zcr.mean()
    df['RMSE'][idx] = rmse.mean()
    df['Chroma'][idx] = chroma_stft.mean()
    df['Centroid'][idx] = spec_cent.mean()
    df['Bandwidth'][idx] = spec_bw.mean()
    df['RollOff'][idx] = rolloff.mean()
    df['Emotion'][idx] = emotion
        
    idx += 1


df = pd.concat([df,mfcc_df])


df.head()


df.to_pickle('feature_dfs/features.pkl')


# mappings
# Anger (ANG)
# Disgust (DIS)
# Fear (FEA)
# Happy/Joy (HAP)
# Neutral (NEU)
# Sad (SAD)
