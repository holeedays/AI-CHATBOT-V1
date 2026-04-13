from django.db import migrations, models
import django.db.models.deletion


def assign_existing_inputs_to_legacy_user(apps, schema_editor):
    User = apps.get_model("chatbot", "User")
    UserInput = apps.get_model("chatbot", "User_Input")

    if not UserInput.objects.filter(user__isnull=True).exists():
        return

    legacy_user = User.objects.create(name="legacy_user", session_key=None)
    UserInput.objects.filter(user__isnull=True).update(user=legacy_user)


class Migration(migrations.Migration):

    dependencies = [
        ("chatbot", "0003_alter_ai_response_user_query"),
    ]

    operations = [
        migrations.CreateModel(
            name="User",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=50)),
                ("session_key", models.CharField(blank=True, db_index=True, max_length=40, null=True)),
            ],
        ),
        migrations.AddField(
            model_name="user_input",
            name="user",
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to="chatbot.user"),
        ),
        migrations.RunPython(assign_existing_inputs_to_legacy_user, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="user_input",
            name="user",
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to="chatbot.user"),
        ),
    ]
