from transformers import AutoFeatureExtractor, SwinForImageClassification
from PIL import Image
from sklearn.decomposition import PCA

image = Image.open("/home/thomas/Downloads/images_128_128/010/0108775015.jpg")

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

# project the outputs to 8 dimensions with pca
pca = PCA(n_components=8)
print(outputs)
# pca.fit_transform(outputs)


