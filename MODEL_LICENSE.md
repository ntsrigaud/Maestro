# Maestro Model Weights: Research-Use-Only License

This license covers the pretrained model weights and inference artifacts distributed with
Maestro (the Hugging Face repository `ntsrigaud/maestro-lstm-hybrid`, and the copies bundled
in the Maestro installers). It does **not** cover the Maestro source code, which is licensed
under Apache-2.0 (see `LICENSE`).

## Why the weights have a separate license

The weights were trained on data governed by third-party terms:

| Dataset | Terms (as published by the dataset owner) |
| ------- | ----------------------------------------- |
| LD-ConGR (Liu et al., CVPR 2022) | "The data is only allowed for academic research." Access requires a signed agreement. |
| Jester (Materzynska et al., ICCVW 2019) | Distributed by Qualcomm under a research-use license. |
| IPN Hand (Benitez-Garcia et al., ICPR 2020) | CC BY 4.0 (attribution required). |

The most restrictive term applies to the weights as a whole.

## Terms

1. **Permitted use:** academic and non-commercial research, evaluation, and personal
   experimentation.
2. **Not permitted:** commercial use, or use in a product or service offered for a fee,
   without separate written permission from the dataset owners and the copyright holder.
3. **Attribution:** keep this notice and cite the Maestro repository (`CITATION.cff`) and
   the datasets above.
4. **Redistribution:** allowed only unmodified and together with this license.
5. **No warranty:** the weights are provided "as is", without warranty of any kind.

This is a plain-language license written by the author, not legal advice. If the weights
are retrained on data that permits broader use, this file will be replaced.
