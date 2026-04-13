from django.db import models

# Identifier for a specific user (and their corresponding chat histories)
class User(models.Model):
    name = models.CharField(max_length=50)
    # for cookie usage
    session_key = models.CharField(max_length=40, null=True, blank=True, db_index=True)

    def __str__(self) -> str:
        return self.name

# Create your models here.
class User_Input(models.Model):
    contents = models.TextField()
    user = models.ForeignKey(
                        User,
                        on_delete=models.CASCADE
                        )

    def __str__(self) -> str:
        return self.contents
    
    def get_user(self) -> User:
        return self.user

class Ai_Response(models.Model):
    contents = models.TextField()

    # we create a 0..1 relationship in the sql: user_input may not have a corresponding ai_reponse
    # due to errors, token runs, etc, that's fine; at least deletion of user_input -> cascades and deletes its child
    # ai response
    user_query = models.OneToOneField(
                            User_Input, 
                            null=True,
                            blank=True,
                            on_delete=models.CASCADE
                            )

    def __str__(self) -> str:
        return self.contents
    
    def user_input(self) -> User_Input | None:
        return self.user_query 
    
