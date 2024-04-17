from speechbrain.pretrained import EncoderClassifier
import torch
import datetime
from tqdm import tqdm
import json
from glob import glob
import os
from setup_logfile import spb_lang_idf

model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="./",  run_opts={"device":"cuda"})

def round_upto_4(values):
    return round(values,4)

def findmax4(first4probs, first4labels):
    lang_codes = []
    lang_scores = []
    for i,j in zip(first4probs, first4labels):
        score = i
        index = j.unsqueeze(0)
    
        text_lab = model.hparams.label_encoder.decode_torch(index)
        lang_codes.append(text_lab)
        lang_scores.append(i)

    return lang_scores, lang_codes

def speechbrain_utils(audio_files):
    ctr = 0
    for i in audio_files:
        try: 
            start = datetime.datetime.now()
            signal = model.load_audio(i)
            emb = model.encode_batch(signal)
            out_prob = model.mods.classifier(emb).squeeze(1)
            probs = out_prob.sort(descending=True)
            
            first4probs, first4labels = findmax4(list(probs[0][0][:4]), list(probs[1][0][:4]))

            end = datetime.datetime.now()
            audio_actual = i.split('/')[-1][:2].lower()

            spb_lang_idf.info(f'Model- ecpa-tdnn, Actual_language- {audio_actual}, Audio_file_name: {i}, Detected_1- {first4labels[0]}, Detected_2- {first4labels[1]}, Detected_others- {(first4labels[2], first4labels[3])}, Probs- {first4probs[:4]}, Response_time- {(end-start).microseconds}')

        except Exception as e:
            ctr = ctr + 1
            print("Invalid audio", i)
    return ctr     

def batch_infer():
    for i in tqdm(range(len(os.listdir('/home/voice-1/prerna/datasets/lid/')))):
        audio_files = glob('/home/voice-1/prerna/datasets/lid/*.wav', recursive=True)
    count = speechbrain_utils(audio_files)
    #l1, l2 = Parallel(n_jobs=os.cpu_count())(delayed(whisper_utils)(f) for f in tqdm(audio_files,leave=False, total=len(audio_files)))

    return f'Total files: {len(audio_files)}. Out of which {len(audio_files) - count} were valid audio files and {count} were invalid.'

print(batch_infer())

