import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO, SAM

input_data = 'data'
output_label = 'labels'


def auto_annotate(data, det_model='yolov8x.pt', sam_model='sam_b.pt', device='', output_dir=None):
    """
    Automatically annotates images using a YOLO object detection model and a SAM segmentation model.

    Args:
        data (str): Path to a folder containing images to be annotated.
        det_model (str, optional): Pre-trained YOLO detection model. Defaults to 'yolov8x.pt'.
        sam_model (str, optional): Pre-trained SAM segmentation model. Defaults to 'sam_b.pt'.
        device (str, optional): Device to run the models on. Defaults to an empty string (CPU or GPU, if available).
        output_dir (str | None | optional): Directory to save the annotated results.
            Defaults to a 'labels' folder in the same directory as 'data'.

    Example:
        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data='ultralytics/assets', det_model='yolov8n.pt', sam_model='mobile_sam.pt')
        ```
    """
    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)

    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f'{data.stem}_auto_annotate_labels'
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    det_results = det_model(data, stream=True, device=device, max_det=1000, conf=0.3)

    for result in det_results:
        class_ids = result.boxes.cls.int().tolist()  # noqa
        if len(class_ids):
            boxes = result.boxes.xyxy  # Boxes object for bbox outputs
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)
            segments = sam_results[0].masks.xyn  # noqa

            with open(f'{str(Path(output_dir) / Path(result.path).stem)}.txt', 'w') as f:
                for i in range(len(segments)):
                    s = segments[i]
                    if len(s) == 0:
                        continue
                    segment = map(str, segments[i].reshape(-1).tolist())
                    f.write(f'{class_ids[i]} ' + ' '.join(segment) + '\n')


start_time = time.time()
# auto_annotate(data=input_data,
#               output_dir=output_label,
#               det_model="models/yolov8n_word_det_v2.pt",
#               sam_model='models/mobile_sam.pt')
print(time.time() - start_time)
# model = YOLO('models/yolov8n_word_det_v2.pt')
# model.predict('data', show=True, max_det=1000, conf=0.1)
image_map = {}
for filename in os.listdir(input_data):
    fileid = os.path.splitext(filename)[0]
    image_map[fileid] = filename


def random_color():
    rgbl = [255, 0, 0]
    random.shuffle(rgbl)
    return tuple(rgbl)


for filename in os.listdir(output_label):
    fileid = os.path.splitext(filename)[0]

    image = cv2.imread(f'{input_data}/{image_map[fileid]}')
    height, width = image.shape[:2]
    with open(f'{output_label}/{filename}', 'r') as f:
        for line in f:
            line = line.strip()
            d = line.split()
            label_idx = d[0]
            points = [(float(d[i]) * width, float(d[i + 1]) * height) for i in range(1, len(d), 2)]
            print(points)
            # break

            # Polygon corner points coordinates
            pts = np.array(points,
                           np.int32)

            pts = pts.reshape((-1, 1, 2))

            isClosed = True

            # Blue color in BGR
            color = random_color()

            # Line thickness of 2 px
            thickness = 2

            image = cv2.polylines(image, [pts],
                                  isClosed, color, thickness)
        cv2.imshow("test", image)
        cv2.waitKey(0)
