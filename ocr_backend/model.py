import logging
import random
from typing import List, Dict, Optional
from urllib.parse import urlparse

import boto3
import requests
from botocore.exceptions import ClientError

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import DATA_UNDEFINED_NAME, get_image_size

logger = logging.getLogger(__name__)


class OcrBackend(LabelStudioMLBase):
    hostname = ""
    access_token = ""

    def __init__(self, project_id: Optional[str] = None):
        super().__init__(project_id)
        self.score_thresh = 0.1

    def _get_image_url(self, task):
        image_url = task['data'].get('ocr') or task['data'].get(DATA_UNDEFINED_NAME)
        print(image_url)

        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            print(r)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3',
                                  endpoint_url='https://s3.tcgroup.vn',
                                  aws_access_key_id='zBQ5GdZLxTkaysvyPxDL',
                                  aws_secret_access_key='MLRGZ5icfd5a7vdkHoe5WPheMD13RyR9fhhdOD9v',
                                  aws_session_token=None,
                                  config=boto3.session.Config(signature_version='s3v4'),
                                  )
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )

            except ClientError as exc:
                logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        print(image_url)
        return image_url

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend)
            :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}''')

        res = []
        for i, task in enumerate(tasks):
            print(f"Task {i}: {task['id']}")
            image_url = self._get_image_url(task)
            if (image_url.startswith("http")):
                image_path = self.get_local_path(image_url)
            else:
                image_path = image_url
            print(image_path)

            url = "https://ocr-core-api.tcgroup.vn/api/v1/ocr/general_with_only_image"

            payload = {}
            files = [
                ('file', (
                    'image_2023-11-10_21-56-25.png',
                    open(image_path, 'rb'),
                    'image/png'))
            ]
            headers = {
                'accept': 'application/json',
                'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0IiwiZXhwIjoxNzA3Nzk1Mjc4fQ.Bv3MTdnBv6nfE517zLhNUjnnLyyqmp2V7pkIu0JMVy0'
            }

            response = requests.request("POST", url, headers=headers, data=payload, files=files)
            print(f"Call api status: {response.status_code}")
            model_results = response.json()
            words = []
            for page in model_results['pages']:
                for block in page['blocks']:
                    for line in block['lines']:
                        for word in line['words']:
                            words.append(word)
            results = []
            all_scores = []
            img_width, img_height = get_image_size(image_path)
            if not words:
                continue
            for word in words:
                output_label = 'Text'
                score = word['confidence']
                if score < self.score_thresh:
                    continue

                # convert the points array from image absolute dimensions to relative dimensions
                x1, y1 = word['bbox'][0]
                x2, y2 = word['bbox'][1]
                x1, x2 = (x1 / img_width) * 100, (x2 / img_width) * 100
                y1, y2 = (y1 / img_height) * 100, (y2 / img_height) * 100

                # must add one for the polygon
                id_gen = str(random.randrange(10 ** 10))
                results.append({
                    'original_width': img_width,
                    'original_height': img_height,
                    'image_rotation': 0,
                    'value': {
                        'x': x1,
                        'y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'rotation': 0
                    },
                    'id': id_gen,
                    'from_name': "bbox",
                    'to_name': 'image',
                    'type': 'rectangle',
                    'origin': 'manual',
                    'score': score,
                })
                # and one for the transcription
                results.append({
                    'original_width': img_width,
                    'original_height': img_height,
                    'image_rotation': 0,
                    'value': {
                        'x': x1,
                        'y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'rotation': 0,
                        "text": [
                            word['value']
                        ]
                    },
                    'id': id_gen,
                    'from_name': "transcription",
                    'to_name': 'image',
                    'type': 'textarea',
                    'origin': 'manual',
                    'score': score,
                })
                # and one for the transcription

                results.append({
                    'original_width': img_width,
                    'original_height': img_height,
                    'image_rotation': 0,
                    'value': {
                        'x': x1,
                        'y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'rotation': 0,
                        "labels": [
                            output_label
                        ]
                    },
                    'id': id_gen,
                    'from_name': "label",
                    'to_name': 'image',
                    'type': 'labels',
                    'origin': 'manual',
                    'score': score,
                })
                all_scores.append(score)
            avg_score = sum(all_scores) / max(len(all_scores), 1)
            res.append({
                'result': results,
                'score': avg_score
            })
        print(res)
        return res

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')
