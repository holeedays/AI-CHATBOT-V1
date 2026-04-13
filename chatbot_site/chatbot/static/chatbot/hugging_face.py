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

from .data_cache import Data_Cache
from ...models import User
import bm25s

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
                user_model: User,
                gemini_model_id: str="gemini-3.1-flash-lite-preview", 
                audio_gen_model_id: str="facebook/musicgen-small",
                image_gen_model_id: str="CompVis/stable-diffusion-v1-4"
                ):
        
        # misc (for tracking users and other stuff like that, keeps a temporary database for each new user)
        # I know this a bit like spaghetti code, but technically this would be the cleanest setup rn
        self.user_model = user_model
        
        # setup our AIs; 
        # although we can link the api key to the settings.py of our django template, this is also a safe bet
        self.gemini_model_id = gemini_model_id
        self.audio_gen_model_id = audio_gen_model_id
        self.image_gen_model_id = image_gen_model_id

        # for RAG retrieval
        self.embedding_model_id = "gemini-embedding-001"

        # loads where our application is running (which is at chatbot_site, the root)
        load_dotenv("test.env")
        self.client = genai.Client(
            api_key=os.getenv("API_KEY")
        )

    # uses gemini to classify the user's input and for possible responses
    def get_input_classification(self, user_input: str) -> dict[Any, Any]:

        # a set of commands in the beginning
        pretext: str = """
            Determine if the following user input fits into these following categories: 
            'music generation, image generation, text generation, miscellaneous'.
            If the input fits into the category to text generation or miscellaneous,
            just respond normally to the user in any appropriate response. If the 
            input fits into music generation or image generation, return the user's input
            exactly as is verbatim.
        """
        # a set of commands at the end (to make sure the ai can provide some context behind their responses)
        sufftext: str = """
            If you categorize the input as either text generation or miscellaneous, use
            the following info to inform your response if there's any:
        """ + self.get_relevant_context(user_input)

        response: genai.types.GenerateContentResponse= self.client.models.generate_content(
            model= self.gemini_model_id, 
            contents=pretext+user_input+sufftext,
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
        
        response = f"""
            Check here '{os.path.abspath(file_path)}' for the sound file. 
            I've also opened the file for you. Check your open tabs!
            """
    
        # this would be the response returned to the user
        return response
    
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

        response = f"""
            Check here '{os.path.abspath(file_path)}' for the image file.
            I've also opened the file for you. Check your open tabs!
            """

        return response

    # RAG retrieval and BM25, not the most ideal or efficient way of caching our context history
    # but its well enough for this example
    def get_relevant_context(self, user_input: str) -> str:
        current_session_context: list[str] = []

        # get our information through our data cache class (makes things slightly neater)
        for input in Data_Cache.get_all_user_inputs(self.user_model):
            u_i = "User Input: " + input.contents

            # there is no block scope compared to java/c# so we can do this
            try:
                # though pylance doesn't know what ai_response.contents is; it should be a str
                ai_r = "Response: " + input.ai_response.contents #type: ignore
            except Data_Cache.ai_response_does_not_exist():
                ai_r = "Response: none"

            # each item of current_session_context essentially represents a single back and forth interction 
            # between the user and the ai (e.g. user response and ai response to that)
            current_session_context.append(u_i + " " + ai_r) #type: ignore

        if current_session_context == []:
            return "No context"

        # implementing relevant searching thru bm25 algorithm ------------------------------------------

        # our context
        corpus_tokens = bm25s.tokenize(current_session_context, stopwords="en", show_progress=False) #type: ignore
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False) #type: ignore

        # user's query
        query_tokens = bm25s.tokenize(user_input, stopwords="en", show_progress=False) #type: ignore
        # bm_results and bm_scores are corresponding with one another
        # bm_results.shape() = (a, b) {axb matrix} where a = # queries (row), b = # of documents/items (column)
        # btw, b is equivalent to the amount specified in the k argument in .retrieve()
        bm_results, bm_scores = retriever.retrieve(query_tokens, k=len(current_session_context), show_progress=False)

        # bm25 only returns scores for the ranked results, so convert them back into a
        # dense per-document score list aligned with current_session_context indices.
        dense_bm25_scores: list[float] = [0.0] * len(current_session_context)
        for result_index, score in zip(bm_results[0], bm_scores[0]): #type: ignore
            dense_bm25_scores[int(result_index)] = float(score)

        # now we're doing dense embedding (RAG retrieval) ----------------------------------------------

        # we're essentially doing the exact same thing except google's api is significantly easier...
        # we can skip tokenization!

        context_embeds = [
            self.extract_embedding_values(
                self.client.models.embed_content(
                    model=self.embedding_model_id,
                    contents=context,
                    config=genai.types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
            )
            for context in current_session_context
        ]

        query_embed = self.extract_embedding_values(
            self.client.models.embed_content(
                model=self.embedding_model_id,
                contents=user_input,
                config=genai.types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
        )

        # get the scores of our embeddings
        embed_scores = [self.cosine_similarity(query_embed, embed) for embed in context_embeds]

        normalized_bm25_scores: list[float] = self.min_max_normalize(dense_bm25_scores)
        normalized_embed_scores: list[float] = self.min_max_normalize(embed_scores)

        # score each document based on an avg of half its lexiconigraphical/keyword similarities (bm25)
        # and half of its semantic similarities (dense embeddings)
        combined_scores: list[float] = [
            # get an average of bm25 scores and embedding scores
            (normalized_bm25_scores[index] + normalized_embed_scores[index]) / 2
            for index in range(len(current_session_context))
        ]

        # np.argmax(iterable) returns the index of the largest value
        # we're returning the current_session_context with the highest avg similarity 
        return current_session_context[int(np.argmax(combined_scores))]
    
    # determines similarities between 2 matrices/tensors (in our case, the vectors of our user_query and context_history)
    @staticmethod
    def cosine_similarity(a: Any, b: Any) -> float:
        a = np.array(a)
        b = np.array(b)
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            return 0.0
        return float(np.dot(a, b) / denominator)

    # normalizes a list of floats (so that all numbers are between 0 and 1)
    @staticmethod
    def min_max_normalize(scores: list[float]) -> list[float]:
        if scores == []:
            print("The list of scores is empty")
            return []

        min_score = min(scores)
        max_score = max(scores)

        if min_score == max_score:
            return [1.0 for _ in scores]

        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]

    # return embedding values (if any)
    @staticmethod
    def extract_embedding_values(embed_response: Any) -> list[float]:
        if hasattr(embed_response, "embeddings") and embed_response.embeddings:
            first_embedding = embed_response.embeddings[0]
            if hasattr(first_embedding, "values"):
                return list(first_embedding.values)
            return list(first_embedding)

        if hasattr(embed_response, "embedding"):
            embedding = embed_response.embedding
            if hasattr(embedding, "values"):
                return list(embedding.values)
            return list(embedding)

        raise ValueError("Unable to read embedding values from response.")
    

    # also manually open the file (beware users, you might get fking jumpscared...)
    @staticmethod
    def open_file_to_user(file_path: str) -> None:
        # windows

        # weird issue opening the file... os.startfile() was yielding winerror 2 (file not found error)
        # even though the relative path existed and the abs path existed as well; finding the abs path seemed to work
        if platform.system() == "Windows":
            os.startfile(os.path.abspath(file_path))  
        # mac
        elif platform.system() == "Darwin":  
            subprocess.call(["open", os.path.abspath(file_path)])
        # linux
        else:  
            subprocess.call(["xdg-open", os.path.abspath(file_path)])
