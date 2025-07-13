from typing import Union
import os
import shutil
from datetime import datetime

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from allosaurus.app import read_recognizer
from pydub import AudioSegment

app = FastAPI()

# Load Allosaurus model
model = read_recognizer("interspeech21")


class NameRequest(BaseModel):
    name: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/submit-name")
def submit_name(request: NameRequest):
    print(f"Received name: {request.name}")
    return {"status": "success", "message": f"Name '{request.name}' received successfully"}


@app.post("/process-audio")
async def process_audio(audio: UploadFile = File(...)):
    try:
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        file_path = os.path.join("audio", filename)
        
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        print(f"Temporary audio file saved: {temp_file_path}")
        
        # Convert to proper WAV format using pydub
        try:
            audio_segment = AudioSegment.from_file(temp_file_path)
            # Export as WAV with specific parameters that Allosaurus expects
            audio_segment.export(file_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])
            print(f"Converted audio file saved: {file_path}")
            
            # Clean up temporary file
            os.remove(temp_file_path)
            
        except Exception as convert_error:
            print(f"Error converting audio: {str(convert_error)}")
            # If conversion fails, try to use the original file
            os.rename(temp_file_path, file_path)
        
        # Process audio through Allosaurus to get IPA characters
        ipa_result = model.recognize(file_path)
        print(f"IPA result: {ipa_result}")
        
        return {
            "status": "success", 
            "message": "Audio processed successfully",
            "filename": filename,
            "file_path": file_path,
            "ipa_result": ipa_result
        }
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return {"status": "error", "message": f"Failed to process audio: {str(e)}"}