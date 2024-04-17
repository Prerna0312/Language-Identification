import whisper
import datetime
import torch
import json
from tqdm import tqdm
import os
from glob import glob
# from setup_logfile import whisper_lang_idf

model = whisper.load_model("base")

with open('languages.json', 'r') as file:
    data = json.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def round_upto_4(values):
    return round(values,4)

def findmax4(p):
    prob_values = []
    labels = []
    for k,v in p.items():
        prob_values.append(v)
    prob_values.sort(reverse=True)

    for i in prob_values[:4]:
        lang_code= ([lang_code for lang_code in p if p[lang_code]==i])
        lang = data.get(str(lang_code[0]))
        labels.append(f'{lang_code[0]}:{lang}')

    return prob_values[:4], labels

def whisper_utils(audio_files):  
    ctr = 0
    for i in audio_files:
        try: 
            start = datetime.datetime.now()        
            audio = whisper.load_audio(i)
            audio = whisper.pad_or_trim(audio)

            mel = whisper.log_mel_spectrogram(audio)
                            
            _, probs = model.detect_language(mel)
            first4probs, first4labels = findmax4(probs)
            end = datetime.datetime.now()
            audio_actual = i.split('/')[-1][:2].lower()
            #whisper_lang_idf.info(f'Model- base, Actual_language- {audio_actual}, Audio file name- {i}, Detected_1- {first4labels[0]}, Detected_2- {first4labels[1]}, Detected_others- {first4labels[2], first4labels[3]}, Probs- {list(map(round_upto_4, first4probs[:4]))}, Response_time- {(end-start).microseconds}')
            # print(list(map(round_upto_4,first4probs)), first4labels, str(audio_actual))
        except Exception as e:
            ctr = ctr + 1
            print("Invalid audio", i)

    return ctr

def batch_infer():
    for i in tqdm(range(len(os.listdir('/home/voice-1/prerna/datasets/lid/')))):
        audio_files = glob('/home/voice-1/prerna/datasets/lid/*.wav', recursive=True)
    count = whisper_utils(audio_files)
    #l1, l2 = Parallel(n_jobs=os.cpu_count())(delayed(whisper_utils)(f) for f in tqdm(audio_files,leave=False, total=len(audio_files)))

    return f'Total files: {len(audio_files)}. Out of which {len(audio_files) - count} were valid audio files and {count} were invalid.'

print(batch_infer())

