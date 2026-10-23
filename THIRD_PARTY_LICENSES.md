# Third-Party Licenses and Attributions

Maestro's source code is licensed under Apache-2.0 (see `LICENSE`).
This file lists third-party material included in or used by Maestro.

## Vendored code

| Component | Location | License | Notes |
| --------- | -------- | ------- | ----- |
| IPN Hand benchmark code (Benitez-Garcia et al., ICPR 2020) | `IPN/` | MIT | Copyright (c) 2020 Gibran Benitez. Original license kept in `IPN/LICENSE`. |

## Runtime dependencies (not vendored)

| Package | License |
| ------- | ------- |
| PySide6 / Qt for Python | LGPL-3.0 (dynamically linked; replaceable in the installer's `_internal` folder) |
| MediaPipe | Apache-2.0 |
| PyTorch | BSD-3-Clause |
| OpenCV (`opencv-python`) | Apache-2.0 |
| NumPy | BSD-3-Clause |
| Pydantic | MIT |
| PyYAML | MIT |
| pyautogui | BSD-3-Clause |
| huggingface_hub | Apache-2.0 |
| PyInstaller (build only) | GPL-2.0 with bootloader exception |

Qt is available from https://www.qt.io/ and https://code.qt.io/. Maestro's installers
include the Qt libraries as separate shared libraries and do not modify them.

## Datasets used to train the released model

The datasets are **not** redistributed in this repository. Their own terms apply to the data
and, as a precaution, to models trained on them (see `MODEL_LICENSE.md`).

| Dataset | Used for | Terms | Reference |
| ------- | -------- | ----- | --------- |
| LD-ConGR | `fist`, swipes | Academic research only; signed agreement required | Liu et al., CVPR 2022 |
| Jester | zoom, `unknown` classes | Qualcomm research-use license | Materzynska et al., ICCVW 2019 |
| IPN Hand | `point_one`, `point_two`, `unknown` | CC BY 4.0 | Benitez-Garcia et al., ICPR 2020 |
