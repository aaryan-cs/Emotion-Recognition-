from flask import Flask, render_template, Response, jsonify
import cv2
from fer import FER
app = Flask(__name__) #flask server initialized

emotion_detector = FER() #fer intitialized

#  Store emotion history for chart(optional)
# emotion_history = {
#     'happy': 0,
#     'sad': 0,
#     'neutral': 0,
#     'angry': 0,
#     'surprise': 0,
#     'disgust': 0,
#     'fear': 0
# }

# openCV function to generate frames to feed to FER
def generate_frames():
    cap = cv2.VideoCapture(0) #initializes the capture feed

    while True:
        success,frame = cap.read()
        if not success:
            break
        else:
            emotions = emotion_detector.detect_emotions(frame) # Analyze emotions on the frame using FER
            # Draw rectangles and labels on the detected faces
            for result in emotions:
                (x, y, w, h) = result['box']
                dominant_emotion = max(result['emotions'], key=result['emotions'].get)

                # Update emotion history for chart(can be added if we want chart to be implemented)
                # emotion_history[dominant_emotion] += 1

                # Define emotion colors
                emotion_colors = {
                    "happy": (0, 255, 0),
                    "sad": (255, 0, 0),
                    "neutral": (255, 255, 255),
                    "angry": (0, 0, 255),
                    "surprise": (255, 255, 0),
                    "disgust": (128, 0, 128),
                    "fear": (255, 165, 0)
                }

                # colour for the emotion most dominant is assigned
                emotion_color = emotion_colors.get(dominant_emotion,(255, 255, 0))

                # face is enclosed in box
                cv2.rectangle(frame,(x, y),(x + w, y + h),emotion_color, 2)

                # Text of emotion on the box
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            emotion_color, 2, cv2.LINE_AA)

            # Encoding the image into jpeg
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Converting jpeg to bytes for browser to render
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/') #homepage route
def index():
    return render_template('index.html')

@app.route('/video_feed') #videofeed route
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/emotion_data')# route for data from fer
# def emotion_data():
#     return jsonify(emotion_history)

if __name__ == "__main__":
    app.run(debug=True)
