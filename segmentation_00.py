from transformers import DetrFeatureExtractor, DetrForSegmentation
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50-panoptic')
model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-50-panoptic')

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
# model predicts COCO classes, bounding boxes, and masks
logits = outputs.logits
bboxes = outputs.pred_boxes
masks = outputs.pred_masks
