import glob
import os.path

import requests

file_pattern = '/data/*.jpg'
project_id = 14
file_key = 'image'
TOKEN = '2f5d2cd5a3531daddfc57fed47e18e18ed671e57'

url = f"172.16.100.204:8200/api/projects/{project_id}/import"


def upload(file_path):
    filename = os.path.basename(file_path)
    payload = {}
    files = [
        (file_key, (filename, open(file_path, 'rb'), 'image/jpeg'))
    ]
    headers = {
        'Authorization': f'Token {TOKEN}',
        'Cookie': 'sessionid=eyJ1aWQiOiJhNjhlMmY5MC1kMDRhLTRiOWQtYTE5MC1mOTM3ZDkxZmM4MTMiLCJvcmdhbml6YXRpb25fcGsiOjF9:1rEMlV:X1eXDoD3MpxYPlFxchWzL1Bqre5cj4hOYhASR4gEOKg'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    if response.status_code == 200:
        print(f"Upload done: {filename}")
    else:
        print(f"Failed upload: {filename}")
    print(response.text)


if __name__ == '__main__':
    for file in glob.glob(file_pattern):
        upload(file)
