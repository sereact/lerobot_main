import time
from typing import Any, Dict

import msgpack


from typing import Any, Dict, List, Union

import numpy as np
import torch


class TorchSerialize:
    def encodes(self, o: Union[torch.Tensor, np.ndarray]) -> dict:
        if isinstance(o, torch.Tensor):
            np_data = o.detach().cpu().numpy()
            return {
                "data": np_data.tobytes(), "dtype": np_data.dtype.str, "encoding": "raw_bytes", "shape": o.shape, "type": "tensor"
            }
        elif isinstance(o, np.ndarray):
            return {
                "data": o.tobytes(), "shape": o.shape, "dtype": o.dtype.str, "encoding": "raw_bytes", "type": "array"
            }
        else:
            return o

    # @timeit(logger)
    def decodes(self, o: Dict) -> Union[torch.Tensor, np.ndarray]:
        dtype = o["dtype"]
        t = o["type"]
        arr = np.frombuffer(o["data"], dtype=dtype)
        arr = arr.reshape(o["shape"])

        if t == "tensor":
            retval = torch.as_tensor(arr)
        elif t == "array":
            retval = arr
        return retval

class JsonLikeSerialize:
    def encodes(self, o: Union[torch.Tensor, np.ndarray, Any]) -> dict:
        if not isinstance(o, list):
            data = o.tolist()
            return {
                "data": data, "dtype": "tensor", "encoding": "json"
            }

img_serializer = TorchSerialize()

class TensorEncoder:

    def __init__(self, image_data_function) -> None:
        self.image_data_function = image_data_function

    def __call__(self, obj: Any) -> Any:
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return self.image_data_function(obj)
        else:
            return obj

class TensorDecoder:
    def __init__(self, image_data_function) -> None:
        self.image_data_function = image_data_function

    def __call__(self, obj: Any) -> Any:
        if '__image_data__' in obj:
            return self.image_data_function(obj)
        else:
            return obj


def encode_image_data(obj: Union[torch.Tensor, np.ndarray]) -> Dict:
    return {"__image_data__": True, "as_str": img_serializer.encodes(obj)}

def encode_image_data_json(obj: Union[torch.Tensor, np.ndarray]) -> List:
    return img_serializer.encodes(obj)

def decode_image_data(obj: Dict) -> Union[torch.Tensor, np.ndarray]:
    return img_serializer.decodes(obj["as_str"])

encoder_data = TensorEncoder(image_data_function=encode_image_data)
decoder_data = TensorDecoder(image_data_function=decode_image_data)
json_like_encoder = TensorEncoder(image_data_function=encode_image_data_json)

def envelope(msg: Any) -> Dict:
    """Wrap a message in an envelope. Just a dict for future json serialization.

    Args:
        msg (Any): Any serializable message, might be (int,string,float,list,dict,tuple)

    Returns:
        dict: Returns the msg wrapped in an envelope.
    """
    return {"payload": msg, "monotonic_time": time.monotonic()}


def rm_envelope(msg: dict) -> Any:
    """Remove the envelope from a message.

    Args:
        msg (dict): The wrapped message

    Returns:
        Any: The payload of the message
    """
    return msg["payload"]


def pack_msg(msg: Any, json_like=False) -> bytes:
    """Packs msg to a bytearray following the msgpack specification.

    Args:
        msg (Any): Any message to send.

    Returns:
        bytearray: [description]
    """
    encoder = json_like_encoder if json_like else encoder_data
    msg = envelope(msg)  # Wrap message in an envelope
    return msgpack.packb(msg, default=encoder, use_bin_type=True)  # Pack to byte array using msgpack


def unpack_msg(packed: bytes, with_header: bool = False) -> Any:
    """Unpack an image message message.

    Args:
        packed (bytearray): bytearray containin a msgpack message
    """
    unpacked = msgpack.unpackb(
        packed, raw=False, object_hook=decoder_data
    )
    if with_header:
        if isinstance(unpacked["monotonic_time"], list):
            unpacked["monotonic_time"] = unpacked["monotonic_time"][0]
        return unpacked
    else:
        return rm_envelope(unpacked)

