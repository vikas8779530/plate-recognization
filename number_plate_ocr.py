
import os
import shutil
import re
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


SOURCE_DIR = "input_images"
DEST_DIR = "OCR_results"
os.makedirs(DEST_DIR, exist_ok=True)


def detect_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if 2 <= w / h <= 6:
                return image[y:y+h, x:x+w]

    return image

def ocr_plate(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    images = [
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 11, 2),
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 11, 2),
    ]

    texts = []
    for img in images:
        for psm in [6, 7, 8]:
            cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            txt = pytesseract.image_to_string(img, config=cfg)
            if txt.strip():
                texts.append(txt)

    return texts

def clean_number(text):
    text = re.sub(r"[^A-Z0-9]", "", text.upper())

    text = text.replace("O", "0").replace("I", "1").replace("Z", "2")
    text = text.replace("W", "M") 
    text = text.replace("ZERO", "ER")

    match = re.search(r"[A-Z]{2}\d{2}[A-Z]{2}\d{4}", text)
    if match:
        return match.group()

    tail = re.search(r"\d{2}[A-Z]{2}\d{4}", text)
    if tail:
        return "MH" + tail.group()

    return None

for img_name in os.listdir(SOURCE_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(SOURCE_DIR, img_name)
    img = cv2.imread(img_path)

    plate_img = detect_plate(img)


    texts = ocr_plate(plate_img)

    success = False
    for txt in texts:
    
        number = clean_number(txt)

        if number:
            shutil.move(img_path, os.path.join(DEST_DIR, f"{number}.jpg"))
            print(f" {img_name} -> {number}")
            success = True
            break

    if not success:
        print(f"{img_name} OCR failed")
