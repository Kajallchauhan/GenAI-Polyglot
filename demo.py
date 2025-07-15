import whisper 

model = whisper.load_model("base") 
result = model.transcribe("Recording.mp3")
print(result["text"])
