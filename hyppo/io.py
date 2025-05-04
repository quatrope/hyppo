from pathlib import Path
import h5py
from hyppo.hsi import HSI

SUPPORTED_LOAD_FORMATS = {".h5"}


def load(path: Path | str):
    path = Path(path)

    file_format = path.suffix
    if file_format not in SUPPORTED_LOAD_FORMATS:
        raise ValueError(f"Unknown Hyper Spectral Image format: {file_format}")

    if file_format == ".h5":
        with h5py.File(path, "r") as f:
            # def ls_dataset(name, node):
            #     if isinstance(node, h5py.Dataset):
            #         print(name, node)

            # f.visititems(ls_dataset)

            reflectance_data = f["SJER/Reflectance/Reflectance_Data"]
            wavelength_data = f["SJER/Reflectance/Metadata/Spectral_Data/Wavelength"]

            data = {"reflectance": reflectance_data, "wavelength": wavelength_data}

    else:
        raise NotImplementedError()

    return HSI(data)


__all__ = ["load", "Path"]
