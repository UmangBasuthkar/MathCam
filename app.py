from flask import Flask, render_template, Response, request, redirect, url_for
from flask_cors import CORS
import cv2 as cv
import numpy as np
import os
import handtracking as ht
from gesture_detection import gen_frames, clear_canvas, save_canvas
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

api_key = os.getenv("GOOGLE_API_KEY").strip()
genai.configure(api_key=api_key)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/gemini")
def gemini():
    # Save the canvas when the solve request is made
    save_canvas()

    if not os.path.exists("saved_canvas.jpg"):
        return "No image found. Please draw something before solving.", 400

    try:
        with open("saved_canvas.jpg", "rb") as f:
            image_data = f.read()
        
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        response = model.generate_content([
            "In the picture you have been provided a mathematical question/equation. Please solve it and give numerical answer. First write the final solution then write the explanation. Give plain text response only.",
            {"mime_type": "image/jpeg", "data": image_data}
        ])
        return response.text
    except Exception as e:
        print(f"Error in Gemini API call: {str(e)}")
        return f"Error processing image: {str(e)}", 500

@app.route("/clear_canvas", methods=["POST"])
def clear_canvas_route():
    clear_canvas()
    return "Canvas cleared", 200

@app.route("/")
def index():
    clear_canvas()  # Clear canvas on page load
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')