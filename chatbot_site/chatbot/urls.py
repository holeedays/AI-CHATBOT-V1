from django.urls import path 
from . import views

urlpatterns = [
    path("", views.chat_page_redirect, name="chat_page_redirect"),
    path("chatbot_session", views.chat_page, name="chatbot_session")
]
