import json
from datetime import datetime
from hashlib import md5
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, model_validator


class MemoryRow(BaseModel):
    """Memory Row that is stored in the database"""

    # id for this memory, auto-generated if not provided
    id: Optional[str] = None
    memory: Dict[str, Any]
    user_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def generate_id(self) -> "MemoryRow":
        if self.id is None:
            from uuid import uuid4
            self.id = str(uuid4())
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude={"created_at", "updated_at"})
        _dict["created_at"] = self.created_at.isoformat() if self.created_at else None
        _dict["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return _dict



class SummaryRow(BaseModel):
    """Session Summary Row that is stored in the database"""

    # id for this memory, auto-generated if not provided
    id: Optional[str] = None
    summary: Dict[str, Any]
    user_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def generate_id(self) -> "MemoryRow":
        if self.id is None:
            from uuid import uuid4
            self.id = str(uuid4())
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude={"created_at", "updated_at"})
        _dict["created_at"] = self.created_at.isoformat() if self.created_at else None
        _dict["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return _dict

