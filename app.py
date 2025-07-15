import os
import whisper
import argostranslate.package, argostranslate.translate
from flask import Flask, request, jsonify, render_template
import json

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static"

# Load Whisper model once
model = whisper.load_model("base")

# Ensure translation model is available or install it
def ensure_translation_installed(from_code, to_code):
    installed_languages = argostranslate.translate.get_installed_languages()
    from_lang = next((lang for lang in installed_languages if lang.code == from_code), None)
    to_lang = next((lang for lang in installed_languages if lang.code == to_code), None)

    if from_lang and to_lang:
        return True

    # Try installing the package
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        (pkg for pkg in available_packages if pkg.from_code == from_code and pkg.to_code == to_code), None
    )

    if package_to_install:
        download_path = package_to_install.download()
        argostranslate.package.install_from_path(download_path)
        return True

    return False
@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == "POST":
        language = request.form["language"]  # ISO code like 'hi', 'fr', etc.
        file = request.files["file"]

        if file:
            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Transcribe with Whisper
            result = model.transcribe(filepath)
            original_text = result["text"]

            # Translate with Argos Translate
            if ensure_translation_installed("en", language):
                installed_languages = argostranslate.translate.get_installed_languages()
                from_lang = next((lang for lang in installed_languages if lang.code == "en"), None)
                to_lang = next((lang for lang in installed_languages if lang.code == language), None)

                if from_lang and to_lang:
                    translation = from_lang.get_translation(to_lang)
                    translated_text = translation.translate(original_text)
                else:
                    translated_text = "Language error after install."
            else:
                translated_text = "Selected language not available."

            return f"""
                <h2>Original Text (English):</h2>
                <p>{original_text}</p>
                <h2>Translated Text ({language}):</h2>
                <p>{translated_text}</p>
                <br><a href="/">Go back</a>
            """
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8080)
