import torch
import numpy as np
from fastapi import FastAPI, Request, Response

from lerobot.utils.messaging import pack_msg, unpack_msg

app = FastAPI()


@app.post("/predict")
async def predict(request: Request):
    data = await request.body()
    obs_input = unpack_msg(data)

    inf_cfg = obs_input.get("inference_config", {})
    dataset_info = obs_input.get("dataset_info", {})
    n_action_steps = inf_cfg.get("n_action_steps", 10)
    action_dim = dataset_info.get("action_dof", 7)

    # Try to infer batch size from any array-like input
    B = None
    for v in obs_input.values():
        if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            if v.ndim >= 1:
                B = int(v.shape[0])
                break

    actions = torch.zeros((B, n_action_steps, action_dim), dtype=torch.float32)

    packed = pack_msg(actions)
    return Response(content=packed, media_type="application/octet-stream")