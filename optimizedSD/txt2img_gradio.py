import gradio as gr
import numpy as np
import torch
from torchvision.utils import make_grid
from einops import rearrange
import os, re
from PIL import Image
import torch
import pandas as pd
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger
from transformers import logging
logging.set_verbosity_error()
import mimetypes
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

# https://github.com/basujindal/stable-diffusion/pull/220/commits/946e2440f557339807a23a4601bb705323dce123
def vectorize_prompt(modelCS, batch_size, prompt):
    empty_result = modelCS.get_learned_conditioning(batch_size * [""])
    result = torch.zeros_like(empty_result)
    subprompts, weights = split_weighted_subprompts(prompt)
    weights_sum = sum(weights)
    cntr = 0
    for i, subprompt in enumerate(subprompts):
        cntr += 1
        result = torch.add(result,
                           modelCS.get_learned_conditioning(batch_size
                                                            * [subprompt]),
                           alpha=weights[i] / weights_sum)
    if cntr == 0:
        result = empty_result
    return result

def load_model_file(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

def load_model(ckpt):
    config = "optimizedSD/v1-inference.yaml"
    sd = load_model_file(f"{ckpt}")
    li, lo = [], []
    for key, v_ in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{config}")

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd

    return (model, modelCS, modelFS)

models_list = {"standard":  load_model("models/sd-latest.ckpt"),
               "finetuned": load_model("models/finetuned-1.ckpt")
               }

def generate(
    model_type,
    prompt,
    ddim_steps,
    n_iter,
    batch_size,
    Height,
    Width,
    scale,
    ddim_eta,
    unet_bs,
    device,
    seed,
    outdir,
    img_format,
    turbo,
    full_precision,
    sampler,
):
    if model_type not in models_list:
        model_type = "standard"

    print(f"using {model_type} model")
    model, modelCS, modelFS = models_list[model_type]

    # negative prompts
    anti_prompts = re.findall(r'\[([^[\]]+)\]', prompt)
    anti_prompts = ", ".join(anti_prompts)
    print(f"negative: {anti_prompts}")

    # clean the prompt
    # remove negative prompts
    prompt = re.sub(r'\[.*]', r'', prompt)

    # remove extra spaces
    prompt = re.sub(r'\s{2,}', r' ', prompt)

    # remove extra commas like ,,
    prompt = re.sub(r',{2,}', r',', prompt)

    # remove possible comma and/or space at end
    prompt = re.sub(r',*\s*$', r'', prompt)

    print(f"cleaned prompt: {prompt}")
    
    C = 4
    f = 8
    start_code = None
    model.unet_bs = unet_bs
    model.turbo = turbo
    model.cdevice = device
    modelCS.cond_stage_model.device = device

    if seed == "":
        seed = randint(0, 1000000)
    seed = int(seed)
    seed_everything(seed)
    # Logging
    logger(locals(), "logs/txt2img_gradio_logs.csv")

    if device != "cpu" and full_precision == False:
        model.half()
        modelFS.half()
        modelCS.half()

    tic = time.time()
    
    # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    assert prompt is not None
    data = [batch_size * [prompt]]

    if full_precision == False and device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    all_samples = []
    seeds = ""
    with torch.no_grad():

        all_samples = list()
        for _ in trange(n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = None
                    if scale != 1.0:
                        uc = vectorize_prompt(modelCS, batch_size, anti_prompts)
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    c = vectorize_prompt(modelCS, batch_size, prompts[0])

                    shape = [batch_size, C, Height // f, Width // f]

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    samples_ddim = model.sample(
                        S=ddim_steps,
                        conditioning=c,
                        seed=seed,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        x_T=start_code,
                        sampler = sampler,
                    )

                    modelFS.to(device)
                    print("uploading images")
                    for i in range(batch_size):

                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        all_samples.append(x_sample.to("cpu"))
                        seeds += str(seed) + ","
                        seed += 1

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    del samples_ddim
                    del x_sample
                    del x_samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    toc = time.time()

    time_taken = (toc - tic)
    grid = torch.cat(all_samples, 0)
    grid = make_grid(grid, nrow=n_iter)
    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()

    txt = (
        "Samples finished in "
        + str(round(time_taken, 3))
        + " seconds"
        + "\nSeeds used = "
        + seeds[:-1]
    )
    return Image.fromarray(grid.astype(np.uint8)), txt


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Text(value="standard"),
        "text",
        gr.Slider(1, 1000, value=50),
        gr.Slider(1, 100, step=1),
        gr.Slider(1, 100, step=1),
        gr.Slider(64, 4096, value=512, step=64),
        gr.Slider(64, 4096, value=512, step=64),
        gr.Slider(0, 50, value=7.5, step=0.1),
        gr.Slider(0, 1, step=0.01),
        gr.Slider(1, 2, value=1, step=1),
        gr.Text(value="cuda"),
        "text",
        gr.Text(value="outputs/txt2img-samples"),
        gr.Radio(["png", "jpg"], value='png'),
        "checkbox",
        "checkbox",
        gr.Radio(["ddim", "plms","heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms"], value="plms"),
    ],
    outputs=["image", "text"],
)

username = os.environ.get("GRADIO_USER")
password = os.environ.get("GRADIO_PASS")
demo.launch(auth=(username, password))
