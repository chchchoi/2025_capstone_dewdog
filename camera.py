from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
import firebase_admin
from firebase_admin import credentials, firestore, storage
from insightface.app import FaceAnalysis
from datetime import datetime

# Firebase 초기화
cred = credentials.Certificate("checkmates-5afa6-firebase-adminsdk-fbsvc-35ff631140.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'checkmates-5afa6.firebasestorage.app'  # ← 본인 Firebase에 맞게 수정
})
db = firestore.client()
bucket = storage.bucket()

# InsightFace 초기화
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Flask 앱
flask_app = Flask(__name__)

def get_embedding(image):
    faces = app.get(image)
    if faces:
        return faces[0].embedding
    return None

def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

@flask_app.route('/register', methods=['POST'])
def register_face():
    print("==== POST /register 요청 도착 ====")

    email = request.form.get("email")  # Unity에서 전송한 이메일
    image_file = request.files.get("image")

    if not email or not image_file:
        return jsonify({"error": "이메일과 이미지가 필요합니다."}), 400

    image_bytes = image_file.read()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "이미지 디코딩 실패"}), 400

    faces = app.get(img)
    if not faces:
        return jsonify({"error": "얼굴을 찾을 수 없습니다."}), 400

    # 파일명을 이메일 기반으로 저장
    blob_path = f"faces/{email}.jpg"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(image_bytes, content_type="image/jpeg")
    blob.make_public()

    print(f"{email} 얼굴 이미지 등록 완료: {blob.public_url}")
    return jsonify({
        "message": f"{email} 등록 완료",
        "image_url": blob.public_url
    }), 200


@flask_app.route('/check', methods=['POST'])
def check_attendance():
    print("==== POST /check 요청 도착 ====")
    
    if 'image' not in request.files:
        return "No image part", 400

    file = request.files['image']
    image_bytes = file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        print("이미지 디코딩 실패")
        return "Invalid image", 400

    # 업로드된 이미지 임베딩 추출
    input_embedding = get_embedding(img)
    if input_embedding is None:
        return "No face detected", 400

    # Firebase Storage에서 모든 얼굴 이미지 다운로드 및 비교
    face_folder = "faces"
    max_similarity = 0.0
    matched_email = None

    blobs = bucket.list_blobs(prefix=face_folder + "/")
    for blob in blobs:
        if not blob.name.endswith('.jpg'):
            continue

        blob_bytes = blob.download_as_bytes()
        ref_np_arr = np.frombuffer(blob_bytes, np.uint8)
        ref_img = cv2.imdecode(ref_np_arr, cv2.IMREAD_COLOR)
        ref_embedding = get_embedding(ref_img)

        similarity = cosine_similarity(input_embedding, ref_embedding)
        print(f"[DEBUG] comparing {blob.name} similarity={similarity:.4f}")

        if similarity > max_similarity and similarity > 0.45:  # 유사도 기준
            max_similarity = similarity
            matched_email = os.path.basename(blob.name).replace(".jpg", "")

    if matched_email:
        # 출석 정보 Firestore에 저장
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db.collection("attendance").add({
            "email": matched_email,
            "timestamp": now,
            "status": "출석"
        })
        print(f"{matched_email} 출석 정보 업로드 완료")
        return jsonify({"email": matched_email, "similarity": max_similarity}), 200
    else:
        return "No matching face found", 404

if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=5050)