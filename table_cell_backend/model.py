import logging
import os
import random
from typing import List, Dict, Optional
from urllib.parse import urlparse

import boto3
import requests
from botocore.exceptions import ClientError
from ultralytics import YOLO

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import DATA_UNDEFINED_NAME, get_image_size

model_name = os.environ.get("MODEL_NAME", 'tabel_cell_yolov8n_v1.pt')

logger = logging.getLogger(__name__)

BASE_URL_DATA = 'http://172.16.100.204:8200'
TOKEN = '2f5d2cd5a3531daddfc57fed47e18e18ed671e57'


def get_image(image_path, output_path):
    print(f'download image: {image_path}')
    url = f'{BASE_URL_DATA}{image_path}'
    headers = {
        'authority': 'labelstudio.tcgroup.vn',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'en-US,en;q=0.9,vi-VN;q=0.8,vi;q=0.7',
        'cache-control': 'no-cache',
        'Authorization': f'Token {TOKEN}',
        'pragma': 'no-cache',
        'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    }
    if os.path.exists(output_path):
        return output_path
    res = requests.get(url, headers=headers)
    print(f'get image: {res.status_code} - {url}, save to {output_path}')
    if res.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(res.content)
    return output_path


class TabelCellBackend(LabelStudioMLBase):
    hostname = ""
    access_token = ""

    def __init__(self, project_id: Optional[str] = None):
        super().__init__(project_id)
        self.score_thresh = 0.1
        # model_name = 'layout_ocr_yolov8n_v4.pt'
        model_path = f'/app/models/{model_name}'
        if not os.path.exists(model_path):
            model_path = f"/Users/tienthien/workspace/tc_group/label-studio-ml-backend/layout_ocr_backend/models/{model_name}"
        self.model = YOLO(model_path)

    def _get_image_url(self, task):
        image_url = task['data'].get('image') or task['data'].get(DATA_UNDEFINED_NAME)
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
            if not os.path.exists(image_path):
                print(f'not found: {image_path}')
                filename = os.path.basename(image_path)
                image_file_path = f'data/{filename}'
                image_path = get_image(image_path, image_file_path)

            model_results = self.model(image_path, conf=0.5)

            results = []
            all_scores = []
            img_width, img_height = get_image_size(image_path)
            for result in model_results:
                boxes = result.boxes.xyxy.cpu().numpy().tolist()
                clses = result.boxes.cls.cpu().numpy().tolist()
                scores = result.boxes.conf.cpu().numpy().tolist()
                name_dict = result.names
                print(name_dict)
                if not boxes:
                    continue
                for cls, box, score in zip(clses, boxes, scores):
                    x1, y1, x2, y2 = box
                    x1, x2 = (x1 / img_width) * 100, (x2 / img_width) * 100
                    y1, y2 = (y1 / img_height) * 100, (y2 / img_height) * 100
                    class_name = name_dict[cls]

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
                            "rectanglelabels": [
                                class_name
                            ]
                        },
                        'id': id_gen,
                        'from_name': "label",
                        'to_name': 'image',
                        'type': 'rectanglelabels',
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
