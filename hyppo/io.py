from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import h5py
import numpy as np
from hyppo.hsi import HSI

SUPPORTED_LOAD_FORMATS = {".h5"}


def load(path: Path | str, reflectance_path: Optional[str] = None, 
         wavelength_path: Optional[str] = None) -> HSI:
    path = Path(path)

    file_format = path.suffix
    if file_format not in SUPPORTED_LOAD_FORMATS:
        raise ValueError(f"Unknown Hyper Spectral Image format: {file_format}")

    if file_format == ".h5":
        return _load_h5(path, reflectance_path, wavelength_path)
    else:
        raise NotImplementedError(f"Format {file_format} not yet implemented")


def _load_h5(path: Path, reflectance_path: Optional[str] = None, 
             wavelength_path: Optional[str] = None) -> HSI:
    with h5py.File(path, "r") as f:
        # Find reflectance and wavelength datasets using heuristics or provided paths
        ref_dataset, ref_path = _find_reflectance_dataset(f, reflectance_path)
        wave_dataset, wave_path = _find_wavelength_dataset(f, wavelength_path)
        
        if ref_dataset is None:
            raise ValueError("Could not find reflectance dataset in H5 file")
        if wave_dataset is None:
            raise ValueError("Could not find wavelength dataset in H5 file")
        
        # Read reflectance data
        reflectance = ref_dataset[:]
        
        # Apply scale factor if present
        scale_factor = ref_dataset.attrs.get("Scale_Factor", [1.0])[0]
        reflectance = reflectance.astype(np.float32) / scale_factor
        
        # Handle null values
        null_value = ref_dataset.attrs.get("Data_Ignore_Value", [-9999.0])[0]
        # Create 2D mask - a pixel is valid if ALL bands are valid
        mask = np.all(reflectance != (null_value / scale_factor), axis=2)
        
        # Read wavelengths
        wavelengths = wave_dataset[:].astype(np.float32)
        
        # Collect metadata
        metadata = {
            "file_path": str(path),
            "reflectance_path": ref_path,
            "wavelength_path": wave_path,
            "scale_factor": scale_factor,
            "null_value": null_value,
        }
        
        # Add any additional attributes from the reflectance dataset
        for key, value in ref_dataset.attrs.items():
            if key not in ["Scale_Factor", "Data_Ignore_Value"]:
                try:
                    # Convert bytes to string if needed
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    elif isinstance(value, np.ndarray) and value.dtype.char == 'S':
                        value = [v.decode('utf-8') if isinstance(v, bytes) else v for v in value]
                    metadata[key] = value
                except:
                    pass
        
        return HSI(reflectance, wavelengths, mask, metadata)


def _find_reflectance_dataset(f: h5py.File, 
                              provided_path: Optional[str] = None) -> Tuple[Optional[h5py.Dataset], str]:
    if provided_path:
        if provided_path in f:
            node = f[provided_path]
            if isinstance(node, h5py.Dataset) and node.ndim == 3:
                return node, provided_path
        raise ValueError(f"Provided reflectance path '{provided_path}' is invalid")
    
    # Heuristic search
    candidates = []
    
    def visitor(name, node):
        if isinstance(node, h5py.Dataset) and node.ndim == 3:
            # Check if path contains reflectance-related keywords
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in ['reflectance', 'reflectancia', 'radiance']):
                # Prioritize reflectance over radiance
                priority = 1 if 'reflectance' in name_lower or 'reflectancia' in name_lower else 2
                candidates.append((priority, name, node))
    
    f.visititems(visitor)
    
    if candidates:
        # Sort by priority and return the best match
        candidates.sort(key=lambda x: x[0])
        return candidates[0][2], candidates[0][1]
    
    return None, ""


def _find_wavelength_dataset(f: h5py.File, 
                             provided_path: Optional[str] = None) -> Tuple[Optional[h5py.Dataset], str]:
    if provided_path:
        if provided_path in f:
            node = f[provided_path]
            if isinstance(node, h5py.Dataset) and node.ndim == 1:
                return node, provided_path
        raise ValueError(f"Provided wavelength path '{provided_path}' is invalid")
    
    # Heuristic search
    candidates = []
    
    def visitor(name, node):
        if isinstance(node, h5py.Dataset) and node.ndim == 1:
            # Check if path contains wavelength-related keywords
            name_lower = name.lower()
            if 'wavelength' in name_lower:
                candidates.append((name, node))
    
    f.visititems(visitor)
    
    if candidates:
        # Return the first match (could be improved with better heuristics)
        return candidates[0][1], candidates[0][0]
    
    return None, ""


__all__ = ["load", "Path"]
