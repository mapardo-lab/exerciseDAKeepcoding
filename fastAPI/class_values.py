from typing import Optional
from pydantic import BaseModel

class VehicleData(BaseModel): 
    number_plate: str
    model: str
    color: Optional[str] = None