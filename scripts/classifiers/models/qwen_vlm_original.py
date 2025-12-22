# %%
import os
import torch
from transformers import AutoProcessor
import torch
from typing import Union
from typing import List
import gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# from qwen_vl_utils import process_vision_info
import sys

sys.path.append("/mnt/l1/code/robo/flow_label/v2label/qwen-vl-utils/src")
from qwen_vl_utils import process_vision_info

# %%
import base64

max_memory = {
    0: "24GB",
    1: "24GB",
}
HFToken = "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"


class Qwen2_5Model:
    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            # torch_dtype=torch.float16,
            torch_dtype=torch.int8,
            device_map="auto",
            max_memory=max_memory,
            cache_dir="/mnt/l1/huggingface",
            trust_remote_code=True,
            token=HFToken,
        )

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            cache_dir="/mnt/l1/huggingface",
            trust_remote_code=True,
            token=HFToken,
        )

    def load_local_image(self, img_path: Union[list, str], target_size=(224, 224)):
        if not img_path:
            raise ValueError("Image path cannot be None")
        try:
            if isinstance(img_path, list):
                imgs = []
                for img in img_path:
                    with open(img, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode(
                            "utf-8"
                        )
                        imgs.append(encoded_string)
                return imgs
            else:
                with open(img_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                return encoded_string
        except Exception as e:
            raise ValueError(f"Error loading image from {img_path}: {e}")
    
    def load_local_image_v2(self, images: List[dict], target_size=(224, 224)):
        if not images:
            raise ValueError("Image path cannot be None")
        try:
            imgs = []
            for img in images:
                with open(img["image_path"], "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode(
                        "utf-8"
                    )
                    img["image_path"] = encoded_string
                    imgs.append(img)
            return imgs
        except Exception as e:
            raise ValueError(f"Error loading images: {e}")

    def predict(self, image_path, question):
        try:
            self.model.eval()
            img = self.load_local_image(image_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"data:image/jpeg;base64,{img}"},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return output_text[0]
        except Exception as e:
            print(f"Error in predict_answer: {str(e)}")
            return ""
    
    def predict_v2(self, sample_images_with_prompts: List[dict], target_image: str, prompt: str):
        try:
            self.model.eval()
            imgs = self.load_local_image_v2(sample_images_with_prompts)
            image = self.load_local_image(target_image)
            system_message = (
                "You are an expert robot observer. "
                "You will be shown a image and must respond with "
                "**exactly one** of these labels: initialize, near, attempt to grab, pushing, hanger, goal state. "
                "Do **not** add any other text."
            )
            system = { "role": "system", "content": system_message
                } 
            samples = []
            for img_data in imgs:
                samples.append(
                    {"type": "image", "image": f"data:image/jpeg;base64,{img_data['image_path']}"},
                )
                samples.append({"type": "text", "text": img_data["text"]})
            user_samples = {"role": "user", "content": samples}
            assistant_blank = {"role": "assistant", "content": ""}
            user_query = {
                    "role": "user",
                    "content": [
                    {"type": "image", "image": f"data:image/jpeg;base64,{image}"},
                    {"type": "text",  "text": prompt}
                ]
            }
            conversation = [
                system,
                user_samples,
                assistant_blank,
                user_query
                ]
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(conversation)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}


            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            del generated_ids, inputs, generated_ids_trimmed, image_inputs
            #torch.cuda.empty_cache()
            gc.collect()
            return output_text
        except Exception as e:
            raise Exception(f"Error in predict v2: {str(e)}")
    def load_multi_prompt_ryan(self,sample_images_with_prompts,prompt):
        self.model.eval()
        imgs = self.load_local_image_v2(sample_images_with_prompts)
        samples = []
        for img_data in imgs:
            samples.append(
                {"type": "image", "image": f"data:image/jpeg;base64,{img_data['image_path']}"},
            )
            samples.append({"type": "text", "text": img_data["text"]})
        self.user_samples = {"role": "user", "content": samples}
        self.assistant_blank = {"role": "assistant", "content": ""}
        system_message = (
                "You are an expert robot observer. "
                "You will be shown a image and must respond"
                "only output single character given example images in the prompt"
            )
        self.system = { "role": "system", "content": system_message
            } 
        self.prompt = prompt
    def predict_v2_ryan(self, encoded_image, prompt: str):
        try:
            self.model.eval()
            # image = self.load_local_image(target_image)
            system = { "role": "system", "content":
                "You are an expert robot observer.  You will be shown a image and must respond with **exactly one** of these labels: initialize, near, attempt to grab, pushing, hanger, goal state.  Do **not** add any other text."
                } 
            
            assistant_blank = {"role": "assistant", "content": ""}
            user_query = {
                    "role": "user",
                    "content": [
                    {"type": "image", "image": f"data:image/jpeg;base64,{encoded_image}"},
                    {"type": "text",  "text": prompt}
                ]
            }
            conversation = [
                system,
                self.user_samples,
                assistant_blank,
                user_query
                ]
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(conversation)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}


            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            del generated_ids, inputs, generated_ids_trimmed, image_inputs
            #torch.cuda.empty_cache()
            gc.collect()
            return output_text
        except Exception as e:
            raise Exception(f"Error in predict v2: {str(e)}")
    
    def predict_v2_2_ryan(self, encoded_image):
        try:
            

            user_query = {
                    "role": "user",
                    "content": [
                    {"type": "image", "image": f"data:image/jpeg;base64,{encoded_image}"},
                    {"type": "text",  "text": self.prompt}
                ]
            }
            
            conversation = [
                self.system,
                self.user_samples,
                self.assistant_blank,
                user_query
                ]
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(conversation)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}


            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            del generated_ids, inputs, generated_ids_trimmed, image_inputs
            #torch.cuda.empty_cache()
            gc.collect()
            return output_text
        except Exception as e:
            raise Exception(f"Error in predict v2: {str(e)}")
    def predict_v3(self, sample_images_with_prompts: List[dict], target_image: str, prompt: str):
        try:
            self.model.eval()
            imgs = self.load_local_image_v2(sample_images_with_prompts)
            image = self.load_local_image(target_image)
            system_message = (
                "You are an expert robomimic simulation robot observer. "
                "You will be shown a image and must respond with "
                "**exactly one** of these labels: initialize, near, attempt to grab, pushing, hanger, goal state. "
                "Do **not** add any other text."
            )
            system = { "role": "system", "content": system_message
                } 
            conversation = [system]
            for img_data in imgs:
                content = []
                content.append(
                    {"type": "image", "image": f"data:image/jpeg;base64,{img_data['image_path']}"},
                )
                content.append({"type": "text", "text": img_data["text"]})
                conversation.append({"role": "user", "content": content})
                conversation.append({"role": "assistant", "content": img_data["text"]})

            user_query = {
                    "role": "user",
                    "content": [
                    {"type": "image", "image": f"data:image/jpeg;base64,{image}"},
                    {"type": "text",  "text": prompt}
                ]
            }
            conversation.append(user_query)
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(conversation)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}


            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            del generated_ids, inputs, generated_ids_trimmed, image_inputs
            #torch.cuda.empty_cache()
            gc.collect()
            return output_text
        except Exception as e:
            raise Exception(f"Error in predict v2: {str(e)}")


    def predict_multiple_images(self, image_paths: list, prompt):
        try:
            self.model.eval()
            imgs = self.load_local_image(image_paths)

            content = []
            for img_data in imgs:
                content.append(
                    {"type": "image", "image": f"data:image/jpeg;base64,{img_data}"}
                )
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            del generated_ids, inputs, generated_ids_trimmed, image_inputs
            #torch.cuda.empty_cache()
            gc.collect()
            return output_text
        except Exception as e:
            raise Exception(f"Error in predict_answer: {str(e)}")

# %%
