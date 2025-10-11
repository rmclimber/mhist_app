from ..service_info import *
from google.cloud import storage
import numpy as np

class ModelHandler:
    def __init__(self, service_info: ServiceInfo):
        self.service_info = service_info
        self.client = storage.Client()
    
    async def load_model(self):
        # Download the desired checkpoint from the GCS bucket
        bucket = self.client.bucket(self.service_info.model_bucket)
        blob = bucket.blob(self.service_info.class_model_path)
        model_checkpoint = blob.download_as_bytes()

        # Assumes model is a Lightning checkpoint and extracts the torch model
        self.model = model_checkpoint  # Replace with actual model loading logic

    async def load_test_data(self):
        # Implement test data loading logic here
        bucket = self.client.bucket(self.service_info.data_bucket)
        img_blob = bucket.blob(self.service_info.imgs_path)
        img_data = img_blob.download_as_bytes()
        label_blob = bucket.blob(self.service_info.labels_path)
        label_data = label_blob.download_as_bytes()
        self.imgs = img_data  # Replace with actual image loading logic
        self.labels = label_data  # Replace with actual label loading logic
    
    async def predict_random(self):
        # Select random image
        random_index = np.random.randint(len(self.imgs))
        img = self.imgs[random_index]
        label = self.labels[random_index]

        # Get prediction
        pred = self.model.predict(img)

        # return prediction and actual label
        return pred, label

    def predict(self, input_data):
        # Implement prediction logic here
        return self.model.predict(input_data)