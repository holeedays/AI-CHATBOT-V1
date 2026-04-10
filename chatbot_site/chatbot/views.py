from django.shortcuts import render

# imports for running our chatbot page
from django.http import JsonResponse, HttpResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# our chatbot class and other misc libs
from ...hugging_face import HF
import re

# Create your views here.
def chat_page(request: HttpRequest) -> HttpResponse:
    if (request.method == "POST"):
        # get our user's input
        user_input = request.POST.get("message", "")

        #instantiate our ai models
        hf = HF()
        classification = hf.get_input_classification(user_input=user_input)

        """ 
        structure of our json schema received from classificaiton is:
            {
                category_name: "either music generation, image generation, text generation, or miscellaneous",
                raw_output: "the response from gemini"            
            }

        """
        # regex for finding keyword "image"
        image_gen_regex = r"/\bimage\b/i"
        # technically, the category name should be absolutely constrained so the keyword "audio" shouldn't pop up
        # regex for finding keword music or audio
        music_gen_regex = r"/\b(music|audio)\b/i"

        message = str()
        if (not re.match(image_gen_regex, classification["category_name"]) is None):
            image_filename = "output.png"
            message = hf.generate_image_from_input(user_input=user_input, file_path=image_filename)
        elif(not re.match(music_gen_regex, classification["category_name"]) is None):
            music_filename = "output.wav"
            message = hf.generate_sound_from_input(user_input=user_input, file_path=music_filename)

        return JsonResponse({"message": message})
    
    return HttpResponse("Hello World")
