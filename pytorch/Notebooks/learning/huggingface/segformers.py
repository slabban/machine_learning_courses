from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests
from transformers import pipeline
import torch
import torch.nn.functional as F

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = feature_extractor(images=image, return_dict=False, return_tensors="pt")
dummy_input = torch.rand([1,3,256,256])
#outputs = model(**inputs)
#logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
model.eval()

traced_model = torch.jit.trace(model, dummy_input)
torch.jit.save(traced_model, "traced_segformer.pt")