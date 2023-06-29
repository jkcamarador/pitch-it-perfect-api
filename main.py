from flask import Flask, jsonify, request
import numpy as np
import librosa, librosa.display
import noisereduce as nr
import joblib
from tensorflow.keras.models import load_model

import warnings

warnings.filterwarnings('ignore')


def input_song(fp, pm, pm2, pa, pa2, w, b, ns, fm):
    x, sr = librosa.load(fp)

    x = nr.reduce_noise(x, sr=sr, thresh_n_mult_nonstationary=ns, freq_mask_smooth_hz=fm)

    onset_frames = librosa.onset.onset_detect(y=x, sr=sr, pre_max=pm, post_max=pm2, pre_avg=pa, post_avg=pa2, wait=w,
                                              backtrack=b)
    onset_times = librosa.frames_to_time(onset_frames)

    return x, onset_times


def generate_partitions(x, onset_times, sr=22050):
    parts = []

    for start in range(len(onset_times)):

        loc_start = int(onset_times[start] * sr)

        if start != len(onset_times) - 1:
            loc_end = int(onset_times[start + 1] * sr)
            note = x[loc_start:loc_end]
        else:
            note = x[loc_start:]

        parts.append(note)

    return parts


def rescale(x, maxdur):
    sr = 22050
    dur = x.shape[0] / sr
    i = dur / maxdur if (dur > maxdur) else maxdur / dur if (dur < maxdur) else 1
    aug = librosa.effects.time_stretch(y=x, rate=i)

    num_zero = (maxdur * sr) - len(aug)
    aug = np.concatenate((aug, np.zeros(int(num_zero))))

    return aug


def extract_features(aug, sr=22050, mfcc=120):
    S = librosa.feature.melspectrogram(y=aug, sr=sr, n_mels=mfcc)
    spec = librosa.power_to_db(S, ref=np.max)
    return spec


def get_prediction(file):
    preds = []

    data, onsets = input_song(file, 1, 1, 1, 1, 11, True, 2, 400)
    parts = generate_partitions(data, onsets)
    rescaled_data = [rescale(part, 0.75) for part in parts]

    for ons in range(len(onsets)):
        sample1 = extract_features(rescaled_data[ons])
        sample = np.expand_dims(sample1, axis=0)
        c = model.predict(sample, verbose=0)

        if c.max() > 0.2:
            t = enc.inverse_transform([np.argmax(c)])[0]
        else:
            t = '_'

        preds.append(t)

    onset = []
    pred = []

    for i in range(1, len(onsets)):
        z = i - 1
        if abs(onsets[z] - onsets[i]) <= 0.4:
            if i == 0:
                z = 1
            else:
                continue

        onset.append(onsets[z])
        pred.append(preds[z])

    # for merging notes time onsets
    # onset.append(onsets[0])
    # pred.append(preds[0])
    # for i in range(1, len(onsets)):
    #     z = i - 1 if onsets[i] - onsets[i-1] <= 0.6 else i
    # print(onset)

    data = {round(onset[x], 2): pred[x] for x in range(len(onset))}

    return data


def get_note(file):
    x, sr = librosa.load(file)
    single_note = nr.reduce_noise(x, sr=sr, thresh_n_mult_nonstationary=2, freq_mask_smooth_hz=400)

    rescaled_data = rescale(single_note, 0.75)

    sample1 = extract_features(rescaled_data)
    sample = np.expand_dims(sample1, axis=0)
    c = model.predict(sample, verbose=0)

    if c.max() > 0.85:
        t = enc.inverse_transform([np.argmax(c)])[0]
    else:
        t = ""

    return t


model = load_model('resnet-best-loss.h5')
enc = joblib.load('labelEncoder.joblib')

app = Flask(__name__)


@app.route('/')
def home():
    return "<center><h1>Pitch it Perfect API</h1></center>"


@app.route('/main', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return "<center><h1>Pitch it Perfect API</h1></center>"

    if request.method == 'POST':
        file = request.files['file']
        filename = 'flutesong.wav'
        file.save(filename)

        if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
            ip_address = request.environ['REMOTE_ADDR']
        else:
            ip_address = request.environ['HTTP_X_FORWARDED_FOR']

        prediction = get_prediction(filename)
        print(f"Request from: {ip_address}, classification was successfuly done on {file.filename}")

        return jsonify(
            predictions=str(prediction),
            status=200,
            info="Prediction Successful"
        )


@app.route('/check-note', methods=['POST', 'GET'])
def check():
    if request.method == 'GET':
        return "<center><h1>Pitch it Perfect API</h1></center>"

    if request.method == 'POST':
        file = request.files['file']
        filename = 'flutenote.wav'
        file.save(filename)

        if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
            ip_address = request.environ['REMOTE_ADDR']
        else:
            ip_address = request.environ['HTTP_X_FORWARDED_FOR']

        prediction = get_note(filename)
        print(f"Request from: {ip_address}, note check on {file.filename}")

        return jsonify(
            predictions=str(prediction),
            status=200,
            info="Prediction Successful"
        )


if __name__ == '__main__':
    app.run(debug=False)
