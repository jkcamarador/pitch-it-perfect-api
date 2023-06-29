# pitch-it-perfect-api

## Description
A ResNet50-based Convolutional Neural Network architecture model was trained to recognize and classify all the musical notes played on a standard flute recorder, from the low C note in the first octave to the high D note in the second octave.

## For Testing

Working API Link for request:
https://pitch-it-perfect-7orwwaazlq-as.a.run.app/main

The model is hosted using Google Cloud Services and Google Cloud Run.

## Sample Request for Python
```
import requests
import json

url = 'https://pitch-it-perfect-7orwwaazlq-as.a.run.app/main'

with open(r'Twinkle Twinkle.mp3', 'rb') as wav:
    files = { "file": wav }

    req = requests.post(url, files=files)

    print(req.status_code)
    print(req.text)

# Pretty Print JSON
obj = json.loads(req.text)
json_formatted_str = json.dumps(obj, indent=4)
print(json_formatted_str)
```

## Additional Information
This API is being utilized by a mobile application to visualize the finger position of the musical notes in a flute recorder. Additional features have also been added. To try it out, you can download and use it on an Android device.
Google Play Store Link: https://play.google.com/store/apps/details?id=com.umak.pip
