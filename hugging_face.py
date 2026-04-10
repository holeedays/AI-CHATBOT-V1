import scipy, torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from transformers import AutoProcessor, MusicgenForConditionalGeneration # ideally we're also going to use an LLM

# I'm using Pylance's static type checker, since HuggingFace's types are dynamic I have to do a lot of #type: ignore
# just letting that out there

from google import genai
from pydantic import BaseModel, Field

from typing import Any

from dotenv import load_dotenv
import os, platform, subprocess
import numpy as np



class Classification(BaseModel):
    category_name: str = Field(description="The name of the category")
    raw_output: str = Field(description="All the text you were going to generate in response to the user's input")

class HF():

    """
    Some general notes about the models:

    gemini requires "pip install -q -U google-genai"
    musicgen requires "pip install transformers scipy torch"
    stable-diffusion requires "pip install diffusers transformers scipy torch invisble_watermark safetensors accelerate"

    ideally, to get pytorch to be able to use "cuda" or device(0), install pytorch from https://pytorch.org/get-started/locally/ 
    pick your build, os, etc and select the most recent version of CUDA, you should have a copy and paste command
    that you can select

    **do note that hugging face is finnicky, some models can't use the most recent transformers/diffuser libraries
    (I'm talking about you AudioLDM2) and some commands are deprecated (you'll probably see in the console) so
    just a quick warning about that when using these AI tools
    """

    def __init__(
                self, 
                gemini_model_id: str="gemini-3.1-flash-lite-preview", 
                audio_gen_model_id: str="facebook/musicgen-small",
                image_gen_model_id: str="CompVis/stable-diffusion-v1-4"
                ):
        # setup our AIs; 
        # although we can link the api key to the settings.py of our django template, this is also a safe bet

        self.gemini_model_id = gemini_model_id
        self.audio_gen_model_id = audio_gen_model_id
        self.image_gen_model_id = image_gen_model_id

        load_dotenv("test.env")
        self.client = genai.Client(
            api_key=os.getenv("API_KEY")
        )

    # uses gemini to classify the user's input and for possible responses
    def get_input_classification(self, user_input: str) -> dict[Any, Any]:

        pretext: str = """
        Determine if the following user input fits into these following categories: 
        'music generation, image generation, text generation, miscellaneous'.
        If the input fits into the category to text generation or miscellaneous,
        just respond normally to the user in any appropriate response. If the 
        input fits into music generation or image generation, return the user's input
        exactly as is verbatim.
        """

        response: genai.types.GenerateContentResponse= self.client.models.generate_content(
            model= self.gemini_model_id, 
            contents=pretext+user_input,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=150,
                response_json_schema=Classification.model_json_schema()
            )
        )

        if (not response.text is None):
            return dict(Classification.model_validate_json(response.text))
        
        return dict()

    # we use musicgen-small because AudioLDM2 wouldn't work :/ for our audio generation
    def generate_sound_from_input(self, user_input: str, file_path: str) -> str: 

        processor = AutoProcessor.from_pretrained(self.audio_gen_model_id) #type: ignore
        model = MusicgenForConditionalGeneration.from_pretrained(self.audio_gen_model_id) #type: ignore

        inputs = processor( #type: ignore
            text=[user_input],
            padding=True,
            return_tensors="pt",
        )

        # initially convert the model and tokenizer of text to gpu if applicable
        if (torch.cuda.is_available()):
            model.to("cuda") #type: ignore
            inputs.to("cuda") #type: ignore

        audio_values = model.generate(**inputs, max_new_tokens=256) #type: ignore
        # since scipy can't read a raw tensor (it needs a numpy array) and pytorch can't automatically convert the value
        # since it doesn't have a .kind() attribute, we have to convert the tensor ourselves
        audio_values_np_arr: np.ndarray[Any] = audio_values.cpu().numpy().squeeze() #type: ignore

        try:
            # save the audio sample
            scipy.io.wavfile.write(file_path, rate=16000, data=audio_values_np_arr) #type: ignore
            self.open_file_to_user(file_path)
        except Exception as e:
            print(f"error generating audio: {e}")
            return "Something went wrong! Please try again"
    
        # this would be the response returned to the user
        return f"Check here {os.path.abspath(file_path)} for the sound file"
    
    # we use stable-diffusion-v1-4 for our image generation
    def generate_image_from_input(self, user_input: str, file_path: str) -> str:

        pipe = StableDiffusionPipeline.from_pretrained(self.image_gen_model_id, torch_dtype=torch.float16) #type: ignore

        if (torch.cuda.is_available()):
            pipe = pipe.to("cuda") #type: ignore

        # access our image from or results
        image = pipe(user_input).images[0]  #type: ignore
        
        try:
            # save the image
            image.save(file_path) #type: ignore
            self.open_file_to_user(file_path)
        except Exception as e:
            print(f"error generating image: {e}")
            return "Something went wrong! Please try again"


        return f"Check here {os.path.abspath(file_path)} for the image file"
    
    # also manually open the file (beware users, you might get fking jumpscared...)
    def open_file_to_user(self, file_path: str) -> None:
        # windows
        if platform.system() == "Windows":
            os.startfile(file_path)  
        # mac
        elif platform.system() == "Darwin":  
            subprocess.call(["open", file_path])
        # linux
        else:  
            subprocess.call(["xdg-open", file_path])
    
