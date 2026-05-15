from flask import Flask, request, jsonify
import face_recognition
import os
from datetime import datetime, time as dt_time
from flask import send_file 
import numpy as np

app = Flask(__name__)

# =========================
# LOAD KNOWN FACES
# =========================

known_encodings = []
known_names = []
known_usn = []

path = "students"

for file in os.listdir(path):
    try:
        img_path = os.path.join(path, file)
        img = face_recognition.load_image_file(img_path)

        encodings = face_recognition.face_encodings(img)

        if len(encodings) == 0:
            print(f"No face found in {file}, skipping...")
            continue

        encoding = encodings[0]

        usn, name = file.split("_")
        name = name.split(".")[0]

        known_encodings.append(encoding)
        known_names.append(name)
        known_usn.append(usn)

        print(f"Loaded: {name} ({usn})")

    except Exception as e:
        print(f"Error loading {file}: {e}")

print("All students loaded successfully")

# =========================
# HOME ROUTE
# =========================

@app.route('/')
def home():
    return "Face Attendance Backend Running"

# =========================
# FACE RECOGNITION API
# =========================

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image received"})

        file = request.files['image']

        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img)
        print("Total faces detected:", len(encodings))
        if len(encodings) == 0:
            return jsonify({"status": "No face found"})

        results = []
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        for face_encoding in encodings:

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.5:   # 🔥 threshold (important)

                name = known_names[best_match_index]
                usn = known_usn[best_match_index]

                #name = known_names[index]
                #usn = known_usn[index]

                # 🔥 CHECK DUPLICATE
                already_marked = False

                if os.path.exists("attendance.csv"):
                    with open("attendance.csv", "r") as f:
                        for line in f:
                            parts = line.strip().split(",")

                            if len(parts) >= 3:
                                recorded_date = parts[0]
                                recorded_usn = parts[2]

                                if recorded_usn == usn and recorded_date == date:
                                    already_marked = True
                                    break

                if not already_marked:
                    with open("attendance.csv", "a") as f:
                        f.write(f"{date},{name},{usn},{time}\n")

                results.append({
                    "name": name,
                    "usn": usn,
                    "status": "Present" if not already_marked else "Already Marked"
                })

            else:
                results.append({
                    "status": "Unknown"
                })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})
# =========================
# 📊 GET ATTENDANCE DATA
# =========================

@app.route('/attendance', methods=['GET'])
def get_attendance():
    if not os.path.exists("attendance.csv"):
        return jsonify([])

    data = []
    with open("attendance.csv", "r") as f:
        for line in f:
            parts = line.strip().split(",")

            if len(parts) == 4:
               date, name, usn, time = parts

            elif len(parts) == 3:
               name, usn, time = parts
               date = "Unknown"   # fallback for old data

            else:
               continue  # skip bad lines
            data.append({
                "date": date,
                "name": name,
                "usn": usn,
                "time": time
            })

    return jsonify(data)

  # 🔥 ADD IMPORT AT TOP

@app.route('/download')
def download():
    return send_file("attendance.csv", as_attachment=True)
# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)