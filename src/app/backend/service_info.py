import os
from dataclasses import dataclass, field

@dataclass
class ServiceInfo:
    class_service_url: str = field(init=False)
    val_service_url: str = field(init=False)
    validate: bool = False

    def __post_init__(self):
        self.class_service_url = os.getenv("CLASS_SERVICE_URL", "http://localhost:8000/classify")
        self.val_service_url = os.getenv("VAL_SERVICE_URL", "http://localhost:8001/validate")