from ...models import User ,User_Input, Ai_Response
from django.db import models
from django.contrib.sessions.models import Session
from django.utils import timezone

from typing import Any

# class for managing/accessing our database
class Data_Cache():

    # get all users currently in session
    @staticmethod
    def get_all_users() -> models.QuerySet[User]:
        return User.objects.all()

    # get all user inputs related to a certain user
    @staticmethod
    def get_all_user_inputs(user_model: User) -> models.QuerySet[User_Input]:
        return User_Input.objects.filter(user=user_model).select_related("ai_response").all()
    
    # get all ai response models related to a certain user
    @staticmethod
    def get_all_ai_responses(user_model: User) -> models.QuerySet[Ai_Response]:
        # double undescores is the syntax that django uses to connect relations together
        # right now: ai_response 0..1 -> user_input Many.1 -> user
        return Ai_Response.objects.filter(user_query__user=user_model).select_related("user_query")

    # returns does not exist error if ai_reponse does not exist (for a user_input)
    @staticmethod
    def ai_response_does_not_exist() -> Any:
        return Ai_Response.DoesNotExist

    # saves user by some name into database and returns the corresponding model
    @staticmethod
    def cache_user(user_name: str, session_key: str | None) -> User:
        u = User(name=user_name, session_key=session_key)
        u.save()
        return u

    @staticmethod
    def get_user_by_id(user_id: int) -> User | None:
        try:
            return User.objects.get(id=user_id)
        except User.DoesNotExist:
            return None

    @staticmethod
    def get_user_by_session(user_id: int | None, session_key: str | None) -> User | None:
        if user_id is None or session_key is None:
            return None

        try:
            return User.objects.get(id=user_id, session_key=session_key)
        except User.DoesNotExist:
            return None

    # save user input to database and return the corresponding model
    @staticmethod
    def cache_user_input(user: User, user_input_contents: str) -> User_Input:
        u_i = User_Input(user=user, contents=user_input_contents)
        u_i.save()
        return u_i

    # save ai response to database and returns corresponding model
    @staticmethod
    def cache_ai_response(user_input_model: User_Input, ai_response_contents: str) -> Ai_Response:
        ai_r = Ai_Response(contents=ai_response_contents, user_query=user_input_model)
        ai_r.save()
        return ai_r
    
    # removes all data from the database
    @staticmethod
    def clear_user_data(user: User) -> None:
        user.delete()

    @staticmethod
    def clear_expired_session_data() -> None:
        active_session_keys = list(
            Session.objects.filter(expire_date__gt=timezone.now()).values_list("session_key", flat=True)
        )

        Session.objects.filter(expire_date__lte=timezone.now()).delete()
        User.objects.exclude(session_key__in=active_session_keys).delete()
        
