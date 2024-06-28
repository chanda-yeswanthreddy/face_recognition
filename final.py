import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import cv2 as cv
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import streamlit as st
import threading
from Google import Create_Service  # Assuming this is a custom function to create Gmail API service

# Constants for Gmail API
CLIENT_SECRET_FILE = '.json'
API_NAME = 'gmail'
API_VERSION = 'v1'
SCOPES = ['https://mail.google.com/']

# Initialize Gmail API service
service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

# List of known face names
known_face_names = ['Elon Musk', 'Faf du Plessis', 'Glenn Maxwell', 'MS Dhoni', 'Tom Cruise', 'Virat Kohli','Sachin Tendulkar']

# List of file paths for known face images
known_face_images = [
    r"D:\facerecog_lib\Elon Musk.jpg",
    r"D:\facerecog_lib\Faf du Plessis.jpg",
    r"D:\facerecog_lib\Glenn Maxwell.jpg",
    r"D:\facerecog_lib\MS Dhoni.jpeg",
    r"D:\facerecog_lib\Tom Cruise.jpg",
    r"D:\facerecog_lib\Virat Kohli.jpg",
    r"D:\facerecog_lib\Sachin Tendulkar.jpeg"
]

# Load the known images and extract face encodings
known_face_encodings = []
for image_path in known_face_images:
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:  # Ensure at least one face was found
        face_encoding = face_encodings[0]
        known_face_encodings.append(face_encoding)
    else:
        print(f"No faces found in the image: {image_path}")

# Ensure we have at least one known face encoding
if not known_face_encodings:
    print("No known faces were loaded. Please check the image files.")
    exit()

# Email dictionary
emails = {
    'Faf du Plessis': '@gmail.com',
    'Virat Kohli': '@gmail.com',
    'Glenn Maxwell': '@gmail.com',
    'Elon Musk': '@gmail.com',
    'MS Dhoni': '@gmail.com',
    'Tom Cruise': '@gmail.com',
    'Sachin Tendulkar': @gmail.com'
}

def recognize_face(img):
    face_locations = face_recognition.face_locations(img,number_of_times_to_upsample=2)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    recognized_names = []

    pil_image = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_image)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.56)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            recognized_names.append(name)

        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 255, 0), outline=(0, 255, 0))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(0, 0, 0))

    del draw
    return np.array(pil_image), recognized_names

def send_email_with_image(service, recipient_email, subject, attachment_path, uploaded_image):
    mimeMessage = MIMEMultipart()
    mimeMessage['to'] = recipient_email
    mimeMessage['subject'] = subject

    # Attach the uploaded image
    _, encoded_image = cv.imencode('.jpg', uploaded_image)
    attached_image = MIMEBase('application', 'octet-stream')
    attached_image.set_payload(encoded_image.tobytes())
    encoders.encode_base64(attached_image)
    attached_image.add_header('Content-Disposition', 'attachment', filename='uploaded_image.jpg')
    mimeMessage.attach(attached_image)

    raw_string = base64.urlsafe_b64encode(mimeMessage.as_bytes()).decode()
    message = service.users().messages().send(
        userId='me',
        body={'raw': raw_string}
    ).execute()
    print(f'Sent message to {recipient_email} with id: {message["id"]}')

# Streamlit app
st.title('Email Image')
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
if uploaded_file is not None:
    image = cv.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    if st.button('Send Email'):
        recognized_image, recognized_names = recognize_face(image)
        for recognized_person in recognized_names:
            recipient_email = emails.get(recognized_person)
            if recipient_email:
                subject = f'Uploaded Image - {recognized_person}'
                send_email_thread = threading.Thread(target=send_email_with_image, args=(service, recipient_email, subject, 'uploaded_image.jpg', recognized_image))
                send_email_thread.start()
            else:
                st.error(f'Recipient email not found for {recognized_person}.')
