from fastapi import FastAPI
import json
from urllib import request
from random import randint

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Aperture Harmonics"}

@app.get("/comfy")
async def get_prompt(prompt: str) -> str:
    # from offical comfyui api docu
    prompt_text = """
    {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 8,
                "denoise": 1,
                "latent_image": [
                    "5",
                    0
                ],
                "model": [
                    "4",
                    0
                ],
                "negative": [
                    "7",
                    0
                ],
                "positive": [
                    "6",
                    0
                ],
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": 8566257,
                "steps": 20
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "v1-5-pruned-emaonly.safetensors"
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "batch_size": 1,
                "height": 512,
                "width": 512
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": [
                    "4",
                    1
                ],
                "text": "masterpiece best quality girl"
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": [
                    "4",
                    1
                ],
                "text": "bad hands"
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": [
                    "3",
                    0
                ],
                "vae": [
                    "4",
                    2
                ]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": [
                    "8",
                    0
                ]
            }
        }
    }
    """

    def queue_prompt(prompt):
        p = {"prompt": prompt}
        data = json.dumps(p).encode('utf-8')
        req = request.Request("http://127.0.0.1:t", data=data)
        request.urlopen(req)

    prompt = json.loads(prompt_text)
    # set the text prompt for our positive CLIPTextEncode
    prompt["6"]["inputs"]["text"] = {prompt}

    # set the seed for our KSampler node
    prompt["3"]["inputs"]["seed"] = randint(1, 1000)

    queue_prompt(prompt)

