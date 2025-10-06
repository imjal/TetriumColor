#!/usr/bin/env python3
"""
Library registry system for managing ink libraries.
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path


class LibraryRegistry:
    """Manages ink library metadata and path resolution."""

    def __init__(self, registry_path: str = "data/inksets/library_registry.json"):
        self.registry_path = registry_path
        self.base_path = Path("data/inksets")
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load the library registry from file."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        """Save the library registry to file."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register_library(self, name: str, csv_path: str, metadata: Optional[Dict] = None):
        """Register a new ink library."""
        if metadata is None:
            metadata = {}

        self.registry[name] = {
            "csv_path": csv_path,
            "metadata": metadata,
            "created": metadata.get("created", "unknown")
        }
        self._save_registry()

    def get_library_path(self, name: str) -> Optional[str]:
        """Get the CSV path for a library name."""
        if name in self.registry:
            return self.registry[name]["csv_path"]
        return None

    def list_libraries(self) -> List[str]:
        """List all registered library names."""
        return list(self.registry.keys())

    def library_exists(self, name: str) -> bool:
        """Check if a library exists."""
        return name in self.registry

    def delete_library(self, name: str) -> bool:
        """Remove a library from the registry (does not delete files).

        Returns True if removed, False if it did not exist.
        """
        if name in self.registry:
            del self.registry[name]
            self._save_registry()
            return True
        return False

    def rename_library(self, old_name: str, new_name: str) -> bool:
        """Rename a library key in the registry.

        Updates the key and, if the csv_path follows the standard pattern
        data/inksets/{name}/{name}-inks.csv (relative path stored), also updates
        the csv_path to point at {new_name}/{new_name}-inks.csv. Files are not moved.

        Returns True on success, False if old_name missing or new_name already exists.
        """
        if old_name not in self.registry or new_name in self.registry:
            return False

        entry = self.registry.pop(old_name)

        # Adjust csv_path if it matches the common pattern "{old_name}/{old_name}-inks.csv"
        try:
            rel_path = entry.get("csv_path", "")
            # Normalize to posix-like separators for comparison
            rel_path_obj = Path(rel_path)
            expected_dir = old_name
            expected_file = f"{old_name}-inks.csv"
            if len(rel_path_obj.parts) >= 2 and rel_path_obj.parts[-2] == expected_dir and rel_path_obj.name == expected_file:
                entry["csv_path"] = str(Path(new_name) / f"{new_name}-inks.csv")
        except Exception:
            # If anything unexpected, keep original path
            pass

        self.registry[new_name] = entry
        self._save_registry()
        return True

    def get_library_metadata(self, name: str) -> Dict:
        """Get metadata for a library."""
        if name in self.registry:
            return self.registry[name].get("metadata", {})
        return {}

    def resolve_library_path(self, name: str) -> str:
        """Resolve the full path to a library CSV file."""
        if name in self.registry:
            path = self.registry[name]["csv_path"]
            if os.path.isabs(path):
                return path
            else:
                return os.path.join(self.base_path, path)

        # Fallback: try to find the library in the standard location
        fallback_path = self.base_path / name / f"{name}-inks.csv"
        if fallback_path.exists():
            return str(fallback_path)

        raise ValueError(f"Library '{name}' not found in registry and no fallback found")

    def auto_discover_libraries(self):
        """Auto-discover libraries in the data/inksets directory."""
        if not self.base_path.exists():
            return

        for lib_dir in self.base_path.iterdir():
            if lib_dir.is_dir():
                # Look for CSV files in the directory
                csv_files = list(lib_dir.glob("*.csv"))
                if csv_files:
                    # Prefer files with the library name
                    preferred = lib_dir / f"{lib_dir.name}-inks.csv"
                    if preferred.exists():
                        csv_path = preferred
                    else:
                        # Look for other common patterns
                        for pattern in ["all_inks.csv", "inks.csv", "*.csv"]:
                            matches = list(lib_dir.glob(pattern))
                            if matches:
                                csv_path = matches[0]
                                break
                        else:
                            csv_path = csv_files[0]

                    # Register if not already registered
                    if lib_dir.name not in self.registry:
                        self.register_library(
                            lib_dir.name,
                            str(csv_path.relative_to(self.base_path)),
                            {"auto_discovered": True}
                        )


# Global registry instance
registry = LibraryRegistry()
