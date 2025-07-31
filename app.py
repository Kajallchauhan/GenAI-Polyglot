import os
import google.generativeai as genai
from flask import Flask, request, render_template, json
from dotenv import load_dotenv
import whisper
from gtts import gTTS
import time
import math

# Load environment variables
load_dotenv()

# --- API Key Setup ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("API key for Google (Gemini) must be set in the .env file.")

# Configure the Gemini client
genai.configure(api_key=GOOGLE_API_KEY)

# --- Flask App Setup ---
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# --- Load Whisper Model ---
print("🔁 Loading Whisper ASR model...")
asr_model = whisper.load_model("base")
print("✅ Whisper model loaded.")

# --- Functions ---
def translate_with_gemini(text, target_lang_code):
    target_lang_full_name = LANGUAGE_MAP.get(target_lang_code, {}).get('name', target_lang_code)
    prompt = f"""
    You are an expert linguist. Your task is to translate the following English text into {target_lang_full_name} and provide a confidence score.
    The English text is: "{text}"
    Your response MUST be a valid JSON object with ONLY two keys:
    1. "translation": A string with the translated text in the NATIVE SCRIPT.
    2. "confidence": An integer (0-100) for your confidence in the translation's accuracy.
    JSON Response:
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned_response_text)
        translation = data.get("translation", "[Translation Error]")
        confidence = data.get("confidence", 0)
        print(f"✅ Gemini Translation: '{translation}', Confidence: {confidence}%")
        return translation, confidence
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"❌ Gemini translation or parsing failed: {e}")
        return f"[Translation Error: {e}]", 0

def generate_audio_with_gtts(text, language_code, file_path):
    try:
        tts = gTTS(text=text, lang=language_code, slow=False)
        tts.save(file_path)
        return True
    except Exception as e:
        print(f"❌ gTTS audio generation failed: {e}")
        return False

# Language Map
LANGUAGE_MAP = {
    'es': {'name': 'Spanish', 'flag': '🇪🇸'}, 'fr': {'name': 'French', 'flag': '🇫🇷'}, 'de': {'name': 'German', 'flag': '🇩🇪'}, 'it': {'name': 'Italian', 'flag': '🇮🇹'}, 'pt': {'name': 'Portuguese', 'flag': '🇵🇹'}, 'ru': {'name': 'Russian', 'flag': '🇷🇺'}, 'ja': {'name': 'Japanese', 'flag': '🇯🇵'}, 'ko': {'name': 'Korean', 'flag': '🇰🇷'}, 'zh-CN': {'name': 'Chinese', 'flag': '🇨🇳'}, 'ar': {'name': 'Arabic', 'flag': '🇸🇦'}, 'hi': {'name': 'Hindi', 'flag': '🇮🇳'}, 'te': {'name': 'Telugu', 'flag': '🇮🇳'}, 'ml': {'name': 'Malayalam', 'flag': '🇮🇳'}, 'ta': {'name': 'Tamil', 'flag': '🇮🇳'}, 'kn': {'name': 'Kannada', 'flag': '🇮🇳'}, 'nl': {'name': 'Dutch', 'flag': '🇳🇱'}, 'sv': {'name': 'Swedish', 'flag': '🇸🇪'}, 'no': {'name': 'Norwegian', 'flag': '🇳🇴'}, 'da': {'name': 'Danish', 'flag': '🇩🇰'}, 'fi': {'name': 'Finnish', 'flag': '🇫🇮'}, 'pl': {'name': 'Polish', 'flag': '🇵🇱'}, 'cs': {'name': 'Czech', 'flag': '🇨🇿'}, 'hu': {'name': 'Hungarian', 'flag': '🇭🇺'}, 'tr': {'name': 'Turkish', 'flag': '🇹🇷'},
}

# --- Routes ---
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/app", methods=["GET", "POST"])
def main_app():
    if request.method == "POST":
        language_code = request.form.get("language")
        if language_code == 'zh': language_code = 'zh-CN'
        file = request.files.get("file")

        if file and language_code:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            
            start_transcribe_time = time.time()
            result = asr_model.transcribe(filepath, verbose=False)
            transcription_duration = time.time() - start_transcribe_time
            original_text = result["text"].strip()
            
            transcription_confidence = 0
            if len(result['segments']) > 0:
                valid_segments = [seg for seg in result['segments'] if seg.get('no_speech_prob', 1) < 0.5]
                if valid_segments:
                    avg_logprob = sum(seg['avg_logprob'] for seg in valid_segments) / len(valid_segments)
                    transcription_confidence = math.exp(avg_logprob) * 100

            start_translate_time = time.time()
            translated_text, translation_confidence = translate_with_gemini(original_text, language_code)
            translation_duration = time.time() - start_translate_time

            start_gtts_time = time.time()
            audio_filename = "tts_output.mp3"
            audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_filename)
            success = generate_audio_with_gtts(translated_text, language_code, audio_path)
            gtts_duration = time.time() - start_gtts_time
            
            translation_confidence_num = int(translation_confidence) if str(translation_confidence).isdigit() else 0
            overall_process_accuracy = (transcription_confidence * 0.4) + (translation_confidence_num * 0.6)
            overall_process_time = transcription_duration + translation_duration + gtts_duration
            
            audio_url = f"/static/{audio_filename}" if success else None
            language_details = LANGUAGE_MAP.get(language_code, {})
            language_full_name = language_details.get('name', language_code.upper())
            language_flag = language_details.get('flag', '🌍')
            
            # ✅ --- THIS IS THE KEY CHANGE --- ✅
            # Create a dictionary containing all the data for the chart.
            # This is much cleaner and safer than injecting individual variables.
            chart_data = {
                "overall_accuracy": overall_process_accuracy,
                "transcription_confidence": transcription_confidence,
                "translation_confidence": translation_confidence_num,
                "overall_time": overall_process_time,
                "transcription_time": transcription_duration,
                "translation_time": translation_duration,
                "gtts_time": gtts_duration
            }

            return render_template("index.html",
                                   original_text=original_text,
                                   translated_text=translated_text,
                                   language_name=language_full_name,
                                   language_flag=language_flag,
                                   audio_url=audio_url,
                                   chart_data=chart_data) # Pass the whole dictionary

    # On a GET request, just render the form with no data.
    return render_template("index.html", original_text=None, chart_data=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)