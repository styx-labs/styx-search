from typing import Type, TypeVar, Optional
from pydantic import BaseModel
from datetime import date

T = TypeVar("T", bound="SerializableModel")


class SerializableModel(BaseModel):
    """Base class for models that need Firestore serialization."""

    def dict(self, *args, **kwargs) -> dict:
        """Convert model to a Firestore-compatible dictionary."""
        d = super().dict(*args, **kwargs)
        return self._serialize_dict(d)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """Create model instance from a Firestore dictionary."""
        if not data:
            return None
        return cls(**cls._deserialize_dict(data))

    @staticmethod
    def _serialize_dict(d: dict) -> dict:
        """Recursively serialize dictionary values."""
        for key, value in d.items():
            if isinstance(value, date):
                d[key] = value.isoformat()
            elif isinstance(value, dict):
                d[key] = SerializableModel._serialize_dict(value)
            elif isinstance(value, list):
                d[key] = [
                    item.dict()
                    if isinstance(item, SerializableModel)
                    else SerializableModel._serialize_dict(item)
                    if isinstance(item, dict)
                    else item.isoformat()
                    if isinstance(item, date)
                    else item
                    for item in value
                ]
        return d

    @staticmethod
    def _deserialize_dict(d: dict) -> dict:
        """Recursively deserialize dictionary values."""
        for key, value in d.items():
            if isinstance(value, str):
                try:
                    d[key] = date.fromisoformat(value)
                except ValueError:
                    pass
            elif isinstance(value, dict):
                d[key] = SerializableModel._deserialize_dict(value)
            elif isinstance(value, list):
                d[key] = [
                    SerializableModel._deserialize_dict(item)
                    if isinstance(item, dict)
                    else date.fromisoformat(item)
                    if isinstance(item, str) and SerializableModel._is_iso_date(item)
                    else item
                    for item in value
                ]
        return d

    @staticmethod
    def _is_iso_date(value: str) -> bool:
        """Check if a string is an ISO format date."""
        try:
            date.fromisoformat(value)
            return True
        except ValueError:
            return False
