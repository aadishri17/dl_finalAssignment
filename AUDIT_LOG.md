# MAI/IDL SS26 - Final Assignment Audit Log

**Date**: June 27 – July 8, 2026
**Author**: Code Audit System
**Codebase Version**: Recovered (Pre-Fix) → Corrected

---

## Executive Summary

**Total Bugs Found**: 14 
**Risk Level**: BLOCKING — Pipeline would crash, silently corrupt gradients, or produce invalid/leaked validation metrics without the fixes below. One infrastructure issue (portability of the config path) remains open and unfixed in the current codebase.

---

## Bug Registry

| Bug ID | File Name | Manifestation | Mathematical / Logical Root Cause | Structural Correction | Git Commit |
|--------|-----------|----------------|-------------------------------------|--------------------------|------------|
| BUG-001 | data.py | Runtime crash (`FileNotFoundError`) when `get_loaders()` tries to load the dataset tensor file | Path construction did not correctly point to the saved `{data}.pt` file naming convention | Corrected path construction to `Path(data_path) / f"{data}.pt"` |  |
| BUG-002 | data.py | Silent logical flaw: validation accuracy/loss appeared artificially strong because the model was evaluated on samples it had already seen in training (data leakage) | Training slice did not exclude the validation portion of `train_images`/`train_labels` before both were used, so train and val sets overlapped | Training data now sliced as `[:val_start]` and validation data as `[val_start:]`, enforcing a strict, non-overlapping hold-out boundary | |
| BUG-003 | data.py (3 occurrences: `train_labels`, `val_labels`, `test_labels`) | Runtime crash / shape mismatch feeding labels into `CrossEntropyLoss` (expects `[N]`, not `[N,1]`) | Labels were stored with an extra singleton dimension (`[N,1]`) inherited from the source `.pt` files | Applied `.squeeze(1)` to all three label tensors to reduce shape to `[N]` |  |
| BUG-004 | models.py (`AlexNet.__init__`) | Runtime shape-mismatch crash in the first convolution layer | Input channel count for the first `Conv2d` was not correctly parameterized against the actual configured `in_channels` | First `Conv2d` now correctly consumes `in_channels` passed in from config |  |
| BUG-005 | models.py (`AlexNet.classifier`) | Runtime crash: `mat1 and mat2 shapes cannot be multiplied` when flattened features hit the first `Linear` layer | Flattened feature dimension was miscalculated; actual output of the conv stack is `192 channels × 4 × 4 = 3072`, not the assumed smaller value | First `Linear` layer input dimension corrected to `3072` to match the true flattened feature-map size |  |
| BUG-006 | models.py (`VGG16.classifier`) | Runtime crash: shape mismatch in the first `Linear` layer of the classifier | Flattened feature dimension miscalculated; VGGBlock stack outputs `512 channels × 4 × 4 = 2048` | First `Linear` layer input dimension corrected to `2048` |  |
| BUG-007 | models.py (`ResNet18.forward`) | Silent logical flaw: `forward()` returned `None`, causing a crash downstream when loss/accuracy tries to operate on the model output | Missing `return` statement at the end of `forward()` — Python implicitly returns `None` | Added explicit `return self.classifier(out)` |  |
| BUG-008 | fit.py (`Trainer.train_one_epoch`) | Numerical/gradient failure: loss explodes or fails to converge across epochs | `optimizer.zero_grad()` was called *after* `loss.backward()`, so gradients from the previous batch accumulated on top of the current batch's gradients every step | Moved `optimizer.zero_grad()` to execute before the forward/backward pass each iteration |  |
| BUG-009 | train.py (`main`) | Runtime crash: `TypeError` — model constructor missing a required keyword argument | `drop_rate` was read from config but never forwarded into the `model_class(...)` call | Added `drop_rate=config["DROP_RATE"]` to the model instantiation |  |
| BUG-010 | fit.py (`Trainer.train_one_epoch`) | Silent anti-pattern: local variable shadowed the Python built-in `sum()`, risking breakage of any later code in scope that relies on the built-in | Variable was named `sum` instead of a non-reserved name | Renamed variable to `num_sum` |  |
| BUG-011 | models.py (`VGGBlock`) | Silent logical flaw: spatial dimensions grew incorrectly through Config-C blocks' 1×1 conv tail, corrupting the feature-map size progression and causing downstream shape errors in the classifier | 1×1 convolutions used `padding=1` (same as the 3×3 convs), which is not "same" padding for a 1×1 kernel | Conditioned `padding = 0 if kernel_size == 1 else 1` so 1×1 convs use zero padding |  |
| BUG-012 | models.py (`VGGBlock`) | Silent logical flaw / eventual crash: only the first conv layer in a multi-conv block received the correct input-channel count; subsequent layers in the same block still expected the block's original `in_channels`, causing a channel mismatch | `current_in_channels` was never updated after appending each conv layer, so all conv layers in a block reused the same (stale) input channel count | Added `current_in_channels = out_channels` update after each conv layer is appended |  |
| BUG-013 | models.py (`AlexNet.classifier`) | Silent logical flaw: model silently fails (shape mismatch or unusable predictions) on any dataset where `num_classes != 11` | Final `Linear` layer's output dimension was hardcoded to `11` instead of being driven by config | Changed final `Linear` layer output to use the `num_classes` parameter from config |  |

---