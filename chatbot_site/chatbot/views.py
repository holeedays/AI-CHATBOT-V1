from django.shortcuts import render

# imports for running our chatbot page
from django.http import JsonResponse, HttpResponse, HttpRequest
# allows us to make POST, PUT, etc requests without a csrf token embed
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# our chatbot class and other misc libs
from .static.chatbot.hugging_face import HF
import re

# Create your views here.
@csrf_exempt
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
        image_gen_regex = r"(?i)image"
        # technically, the category name should be absolutely constrained so the keyword "audio" shouldn't pop up
        # regex for finding keword music or audio
        music_gen_regex = r"(?i)(music|audio)"

        message = str()
        if (not re.match(image_gen_regex, classification["category_name"]) is None):
            image_filename = "temp/output.png"
            message = hf.generate_image_from_input(user_input=user_input, file_path=image_filename)
        elif(not re.match(music_gen_regex, classification["category_name"]) is None):
            music_filename = "temp/output.wav"
            message = hf.generate_sound_from_input(user_input=user_input, file_path=music_filename)
        else:
            message = classification["raw_output"]

        return JsonResponse({"message": message})
    
    return render(request, "chatbot/chat.html")
