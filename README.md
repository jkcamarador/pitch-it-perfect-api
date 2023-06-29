# pitch-it-perfect-api

Working API Link for request:
https://pitch-it-perfect-7orwwaazlq-as.a.run.app/main

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
