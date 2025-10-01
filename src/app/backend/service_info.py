import os
from dataclasses import dataclass, field

@dataclass
class ServiceInfo:
    # container specifications
    class_service_url: str = field(init=False)
    image_size: int = 224
    val_service_url: str = field(init=False)
    validate: bool = False

    # model info
    model_bucket: str = 'mhist-models'
    class_model_path: str = "models/mhist_model.pth"
    class_labels_path: str = "models/mhist_labels.json"
    val_model_path: str = "models/mhist_val_model.pth"
    val_labels_path: str = "models/mhist_val_labels.json"

    def __post_init__(self):
        self.class_service_url = os.getenv("CLASS_SERVICE_URL", "http://localhost:8000/classify")
        self.val_service_url = os.getenv("VAL_SERVICE_URL", "http://localhost:8001/validate")