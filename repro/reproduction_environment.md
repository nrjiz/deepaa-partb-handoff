# Reproduction environment

This is the environment used for the Gate 0 evaluation on 2026-09-03. It is not claimed to be the exact historical conference environment; no historical lock file exists.

- Virtual environment: `C:\Users\UserPc\Desktop\ai\.venvs\deepaa_data2026_repro` (outside the repository)
- Python: 3.13.3, CPython, AMD64
- OS: Windows 11, build 26200 (`Windows-11-10.0.26200-SP0`)
- Processor string: Intel64 Family 6 Model 142 Stepping 10
- TensorFlow: 2.21.0
- Keras: 3.15.1
- NumPy: 2.5.2
- pandas: 3.0.5
- Matplotlib: 3.11.1
- Pillow: 12.3.0 (transitive Matplotlib dependency; not directly imported by the baseline)
- h5py: 3.14.0 (checkpoint transport; not directly imported by the baseline)
- pip: 25.0.1
- Device used: CPU only; TensorFlow reported `/physical_device:CPU:0`
- scikit-learn, OpenCV, and SciPy: not installed in this isolated environment and not required by the conference script or the reproduction utilities
- Complete installed-package freeze: `requirements_reproduction.txt`

The pre-existing global Python 3.13.3 environment was inspected before isolation. It had no TensorFlow and had NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.8.0, Matplotlib 3.10.3, Pillow 11.3.0, OpenCV 4.12.0, h5py 3.16.0, and SciPy 1.16.3. It was not modified or used for checkpoint evaluation.

## Direct runtime requirements

The authoritative script directly imports NumPy, pandas, Matplotlib, and TensorFlow. Standard-library imports are `os`, `math`, `argparse`, `dataclasses`, `datetime`, and `pathlib`. The evaluation utilities additionally use h5py indirectly through Keras HDF5 weight loading. Pytest was installed for optional future invocation, although the baseline invariant suite is a standalone script.

## Environment creation

```powershell
python -m venv C:\Users\UserPc\Desktop\ai\.venvs\deepaa_data2026_repro
& 'C:\Users\UserPc\Desktop\ai\.venvs\deepaa_data2026_repro\Scripts\python.exe' -m pip install -r repro\requirements_reproduction.txt
```

TensorFlow emitted its expected native-Windows CPU warning and oneDNN notice. No GPU was used. The checkpoint reproduction matched all committed metrics within `1e-5`, with maximum observed difference below `2e-8`.
