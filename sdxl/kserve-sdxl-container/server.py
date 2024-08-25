import argparse
import base64
import io
import os
from typing import Dict, Union

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from kserve import InferRequest, InferResponse, Model, ModelServer, model_server
from kserve.errors import InvalidInput


class DiffusersModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.model_id = args.model_id or "/mnt/models"
        self.refiner_id = args.refiner_id or None
        self.pipeline = None
        self.refiner = None
        self.ready = False
        self.load()

    def load(self):
        # Load the model
        if args.single_file_model and args.single_file_model != "":
            pipeline = StableDiffusionXLPipeline.from_single_file(
                args.single_file_model,
                torch_dtype=torch.float16,
                variant="fp16",
                safety_checker=None,
                use_safetensors=True,
            )
        else:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                safety_checker=None,
                use_safetensors=True,
            )
        if args.device:
            print(f"Loading model on device: {args.device}")
            if args.device == "cuda":
                pipeline.to(torch.device("cuda"))
            elif args.device == "cpu":
                pipeline.to(torch.device("cpu"))
            elif args.device == "enable_model_cpu_offload":
                pipeline.enable_model_cpu_offload()
            elif args.device == "enable_sequential_cpu_offload":
                pipeline.enable_sequential_cpu_offload()
            else:
                raise ValueError(f"Invalid device: {args.device}")
        else:
            pipeline.to(torch.device("cuda"))
        self.pipeline = pipeline

        # Load the refiner model
        if args.use_refiner:
            if args.refiner_single_file_model and args.refiner_single_file_model != "":
                refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    args.refiner_single_file_model,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    safety_checker=None,
                    use_safetensors=True,
                    text_encoder_2=pipeline.text_encoder_2,
                    vae=pipeline.vae,
                )
            else:
                refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    self.refiner_id,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    safety_checker=None,
                    use_safetensors=True,
                )
            if args.device:
                print(f"Loading refiner model on device: {args.device}")
                if args.device == "cuda":
                    refiner.to(torch.device("cuda"))
                elif args.device == "cpu":
                    refiner.to(torch.device("cpu"))
                elif args.device == "enable_model_cpu_offload":
                    refiner.enable_model_cpu_offload()
                elif args.device == "enable_sequential_cpu_offload":
                    refiner.enable_sequential_cpu_offload()
                else:
                    raise ValueError(f"Invalid device: {args.device}")
            else:
                refiner.to(torch.device("cuda"))
            self.refiner = refiner

        # The ready flag is used by model ready endpoint for readiness probes,
        # set to True when model is loaded successfully without exceptions.
        self.ready = True

    def preprocess(
        self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Dict:
        if isinstance(payload, Dict) and "instances" in payload:
            headers["request-type"] = "v1"
        elif isinstance(payload, InferRequest):
            raise InvalidInput("v2 protocol not implemented")
        else:
            raise InvalidInput("invalid payload")

        return payload["instances"][0]

    def convert_lists_to_tuples(self, data):
        if isinstance(data, dict):
            return {k: self.convert_lists_to_tuples(v) for k, v in data.items()}
        elif isinstance(data, list):
            return tuple(self.convert_lists_to_tuples(v) for v in data)
        else:
            return data

    def predict(
        self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Union[Dict, InferResponse]:
        payload = self.convert_lists_to_tuples(payload)

        # Create the image, without refiner if not needed
        if not args.use_refiner:
            image = self.pipeline(**payload).images[0]
        else:
            denoising_limit = payload["denoising_limit"]
            image = self.pipeline(**payload, output_type="latent", denoising_end=denoising_limit).images
            image = self.refiner(**payload,image=image, denoising_start=denoising_limit).images[0]
        
        # Convert the image to base64
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        im_b64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

        return {
            "predictions": [
                {
                    "model_name": self.model_id,
                    "prompt": payload["prompt"],
                    "image": {"format": "PNG", "b64": im_b64},
                }
            ]
        }


parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--model_id",
    type=str,
    help="Model ID to load (default: /mnt/models, adapt if you use the refiner model)",
)
parser.add_argument(
    "--single_file_model", type=str, help="Name of a single file model to load"
)
parser.add_argument("--use_refiner", type=bool, default=False, help="Use the refiner model")
parser.add_argument(
    "--refiner_id",
    type=str,
    help="Refiner model ID to load (or adapt from /mnt/models)",
)
parser.add_argument(
    "--refiner_single_file_model",
    type=str,
    help="Name of a single file refiner model to load",
)
parser.add_argument(
    "--device",
    type=str,
    help="Device to use, including offloading. Valid values are: 'cuda' (default), 'enable_model_cpu_offload', 'enable_sequential_cpu_offload', 'cpu' (works but unusable...)",
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = DiffusersModel(args.model_name)
    model.load()
    ModelServer().start([model])
