from dataclasses import dataclass

@dataclass
class GCPInfo:
    data_bucket: str
    configs_bucket: str
    output_bucket: str
