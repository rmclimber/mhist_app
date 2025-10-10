import os
from dataclasses import dataclass, field

@dataclass
class ServiceInfo:
    ## GCS bucket information
    model_bucket: str = ""
    data_bucket: str = ""

    # GCS paths
    val_model_path: str = ""
    class_model_path: str = ""
    imgs_path: str = ""
    labels_path: str = ""
    data_info_path: str = ""

    # data miscellany
    input_shape: int | tuple
    
    # service configurations
    gateway_service_host: str = ""
    gateway_service_port: int = 0
    class_service_host: str = ""
    class_service_port: int = 0
    val_service_host: str = ""
    val_service_port: int = 0

    # general app configurations
    http_timeout: int = 10
    max_retries: int = 3
    retry_delay: int = 2  # seconds