from django.shortcuts import render, redirect

# imports for running our chatbot page
from django.http import JsonResponse, HttpResponse, HttpResponseBadRequest, HttpRequest
# allows us to make POST, PUT, etc requests without a csrf token embed
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# our chatbot class and other misc libs
from .static.chatbot.hugging_face import HF
from .static.chatbot.data_cache import Data_Cache
import re

# Create your views here.
@csrf_exempt
def chat_page(request: HttpRequest) -> HttpResponse:
    Data_Cache.clear_expired_session_data()

    if request.session.session_key is None:
        request.session.create()

    # get our unique user model here
    current_user = Data_Cache.get_user_by_session(
        user_id=request.session.get("chatbot_user_id"),
        session_key=request.session.session_key
    )

    # if no cookie exists, just terminate the browser 
    if (current_user is None):
        return HttpResponseBadRequest()

    if (request.method == "POST"):
        # get our user's input
        user_input = request.POST.get("message", "")

        #instantiate our ai models
        hf = HF(user_model=current_user)
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

        # cache our data into our database
        u_i = Data_Cache.cache_user_input(current_user, user_input_contents=user_input)
        Data_Cache.cache_ai_response(user_input_model=u_i, ai_response_contents=message)

        return JsonResponse({"message": message})

    return render(request, "chatbot/chat.html")

@csrf_exempt
def chat_page_redirect(request: HttpRequest) -> HttpResponse:
    # clear any expired data
    Data_Cache.clear_expired_session_data()

    # if the cookie already exists, remove existing data
    current_user = Data_Cache.get_user_by_session(
        user_id=request.session.get("chatbot_user_id"),
        session_key=request.session.session_key
    )
    # clear current user data if they somehow get to this page and they still have an active cookie
    if (not current_user is None):
        Data_Cache.clear_user_data(current_user)
        request.session.flush()
    # if there is no cookie, then create one
    if request.session.session_key is None:
        request.session.create()

    # create a cookie that stores each unique instance of users (this allows us to retrieve data and histories for
    # each user uniquely to power our chatbot talks)
    user_name = "user_" + str(Data_Cache.get_all_users().count())
    user = Data_Cache.cache_user(user_name=user_name, session_key=request.session.session_key)
    request.session["chatbot_user_id"] = user.id # type: ignore

    # then just redirect to the main chatbot page
    return redirect("/chatbot_session")

    
