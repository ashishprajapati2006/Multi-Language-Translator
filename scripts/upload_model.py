from huggingface_hub import upload_folder

upload_folder(
    folder_path="models",   # upload entire models folder
    repo_id="ashishprajapati2006/translator-model",
    repo_type="model",
)

print("All models uploaded successfully!")
