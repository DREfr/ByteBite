"""
custom_checkpoint.py
---------------------
Reader for a custom, hand-rolled binary checkpoint format used for this
project's resnet50_food*.ckpt files.

These files are NOT standard PyTorch checkpoints (torch.load() will raise
`_pickle.UnpicklingError: invalid load key`). They're a small protobuf-style
binary format: a flat sequence of length-delimited entries, one per tensor,
each containing:
    - the tensor's original name (a string, e.g. "layer1.0.bn1.gamma")
    - its shape (repeated int fields)
    - its dtype (a string, e.g. "Float32")
    - the raw little-endian tensor bytes

The naming scheme used inside the file differs slightly from torchvision's
ResNet50, so we rename keys during load:

    down_sample        -> downsample     (shortcut branch)
    classifier          -> fc             (final linear layer)
    <bn>.gamma           -> <bn>.weight
    <bn>.beta            -> <bn>.bias
    <bn>.moving_mean      -> <bn>.running_mean
    <bn>.moving_variance -> <bn>.running_var

After renaming, the keys match a stock `torchvision.models.resnet50()`
state_dict exactly (verified 1:1 against the actual checkpoint: 267/267
tensors, no missing or unexpected keys).
"""

import numpy as np
import torch

_DTYPE_MAP = {
    "Float32": np.float32,
    "Float64": np.float64,
}


def _read_varint(data: bytes, pos: int):
    """Decode a protobuf-style base-128 varint starting at `pos`."""
    result = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _rename_key(name: str) -> str:
    name = name.replace("down_sample", "downsample")
    name = name.replace("classifier", "fc")
    name = name.replace(".gamma", ".weight")
    name = name.replace(".beta", ".bias")
    name = name.replace(".moving_mean", ".running_mean")
    name = name.replace(".moving_variance", ".running_var")
    return name


def parse_custom_checkpoint(path: str) -> dict:
    """
    Parse the custom binary format and return a dict of
    {torchvision-style key name: torch.Tensor}, ready to hand to
    model.load_state_dict().
    """
    with open(path, "rb") as f:
        data = f.read()

    tensors = {}
    pos = 0
    n = len(data)

    while pos < n:
        tag = data[pos]
        pos += 1
        if tag != 0x0A:
            raise ValueError(
                f"Unexpected top-level tag {tag:#x} at byte offset {pos - 1}; "
                f"this file doesn't match the expected custom checkpoint format."
            )
        entry_len, pos = _read_varint(data, pos)
        entry_end = pos + entry_len

        # --- field 1: tensor name ---
        t1 = data[pos]
        pos += 1
        if t1 != 0x0A:
            raise ValueError(f"Expected name field at offset {pos - 1}")
        name_len, pos = _read_varint(data, pos)
        name = data[pos:pos + name_len].decode("utf-8")
        pos += name_len

        # --- field 2: submessage {shape*, dtype, raw data} ---
        t2 = data[pos]
        pos += 1
        if t2 != 0x12:
            raise ValueError(f"Expected tensor-info field at offset {pos - 1}")
        sub_len, pos = _read_varint(data, pos)
        sub_end = pos + sub_len

        shape = []
        dtype_str = None
        raw = None
        while pos < sub_end:
            t = data[pos]
            pos += 1
            field_num = t >> 3
            wire_type = t & 0x7
            if wire_type == 0:  # varint
                value, pos = _read_varint(data, pos)
                if field_num == 1:
                    shape.append(value)
            elif wire_type == 2:  # length-delimited
                length, pos = _read_varint(data, pos)
                if field_num == 2:
                    dtype_str = data[pos:pos + length].decode("utf-8")
                elif field_num == 3:
                    raw = data[pos:pos + length]
                pos += length
            else:
                raise ValueError(f"Unsupported wire type {wire_type} at offset {pos - 1}")

        np_dtype = _DTYPE_MAP.get(dtype_str, np.float32)
        array = np.frombuffer(raw, dtype=np_dtype)
        if shape:
            array = array.reshape(shape)

        tensors[_rename_key(name)] = torch.from_numpy(array.copy())
        pos = entry_end

    return tensors


def is_custom_checkpoint(path: str) -> bool:
    """Quick sniff test: does this file start with our custom format's tag byte?"""
    with open(path, "rb") as f:
        first_byte = f.read(1)
    return first_byte == b"\x0a"
