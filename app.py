import io
import json

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import batched_nms
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request


app = Flask(__name__)

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
DETECTION_URL = '/predict'
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5


def model_predict(model, image):
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    pred = model(tensor)

    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = list(pred[0]['boxes'].detach().numpy())
    pred_score = list(pred[0]['scores'].detach().numpy())

    indices = batched_nms(pred[0]['boxes'], pred[0]['scores'], pred[0]['labels'], NMS_THRESHOLD).numpy()
    pred_idexs = [i for i, score in enumerate(pred_score) if score > SCORE_THRESHOLD and i in indices]

    pred_class = [pred_class[i] for i in pred_idexs]
    pred_boxes = [pred_boxes[i] for i in pred_idexs]
    pred_score = [pred_score[i] for i in pred_idexs]

    return pred_class, pred_boxes, pred_score


def make_response(pred_class, pred_boxes, pred_score):
    count_objects = len(pred_class)
    objects = []
    for i in range(count_objects):
        object_dict = {
            'ObjectClassName': pred_class[i],
            'Height': round(pred_boxes[i][2]) - round(pred_boxes[i][0]),
            'Width': round(pred_boxes[i][3]) - round(pred_boxes[i][1]),
            'Score': float(pred_score[i]),
            'X': round(pred_boxes[i][0]),
            'Y': round(pred_boxes[i][1]),
        }
        objects.append(object_dict)

    response = {
        'Successful': count_objects > 0,
        'Objects': objects,
        'ObjectCount': count_objects,
    }
    return json.dumps(response)


@app.route(DETECTION_URL, methods=['POST'])
def predict():
    if not request.method == 'POST':
        return 'Request type must be POST.', 400

    if not request.files.get('imageFile'):
        return '"imageFile" not found in request parameters.', 400

    image_file = request.files['imageFile']
    image_bytes = image_file.read()

    image = Image.open(io.BytesIO(image_bytes))

    pred_class, pred_boxes, pred_score = model_predict(model, image)

    return make_response(pred_class, pred_boxes, pred_score), 200


if __name__ == '__main__':
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    app.run(host='0.0.0.0', port=5555)
