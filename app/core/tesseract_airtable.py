from flask import Flask, request, jsonify
import cv2
import numpy as np
import pytesseract
import re
from datetime import datetime
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import io
import base64
import requests
import json
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)

# Configuration Airtable
AIRTABLE_API_KEY = os.getenv('AIRTABLE_API_KEY')
AIRTABLE_BASE_ID = os.getenv('AIRTABLE_BASE_ID')
AIRTABLE_RECEIPTS_TABLE = os.getenv('AIRTABLE_RECEIPTS_TABLE', 'Receipts')
AIRTABLE_ITEMS_TABLE = os.getenv('AIRTABLE_ITEMS_TABLE', 'Items')


# Configuration Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Décommentez et ajustez pour Windows

@dataclass
class ReceiptItem:
    description: str
    quantity: float
    unit_price: float
    total_price: float


@dataclass
class ReceiptData:
    store_name: Optional[str] = None
    date: Optional[datetime] = None
    time: Optional[str] = None
    total_amount: Optional[float] = None
    tax_amount: Optional[float] = None
    payment_method: Optional[str] = None
    items: List[ReceiptItem] = None

    def __post_init__(self):
        if self.items is None:
            self.items = []


def preprocess_image(image_data):
    """
    Preprocess the receipt image to enhance text recognition

    Args:
        image_data: Image data as numpy array
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Noise removal with morphological operations
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Dilation to make text more visible
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(opening, kernel, iterations=1)

    # Apply additional adaptive thresholding for better results
    adaptive_thresh = cv2.adaptiveThreshold(
        dilated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return adaptive_thresh


def extract_text(preprocessed_img):
    """
    Extract text from preprocessed image using Tesseract OCR
    """
    # Configure Tesseract parameters
    custom_config = r'--oem 3 --psm 6'

    # Extract text
    text = pytesseract.image_to_string(preprocessed_img, config=custom_config)

    return text


def parse_receipt(text):
    """
    Parse the extracted text into structured receipt data
    """
    receipt = ReceiptData()
    lines = text.strip().split('\n')

    # Store name is typically at the top
    if lines and lines[0].strip():
        receipt.store_name = lines[0].strip()

    # Look for date and time
    date_pattern = r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})'
    time_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)'

    for line in lines:
        # Date extraction
        date_match = re.search(date_pattern, line)
        if date_match and not receipt.date:
            date_str = date_match.group(1)
            try:
                # Try different date formats
                for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y', '%d/%m/%y', '%m/%d/%y']:
                    try:
                        receipt.date = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
            except:
                pass

        # Time extraction
        time_match = re.search(time_pattern, line)
        if time_match and not receipt.time:
            receipt.time = time_match.group(1)

        # Total amount extraction
        total_match = re.search(r'TOTAL[\s:]*\$?(\d+\.\d{2})', line, re.IGNORECASE)
        if total_match and not receipt.total_amount:
            try:
                receipt.total_amount = float(total_match.group(1))
            except:
                pass

        # Tax amount extraction
        tax_match = re.search(r'TAX[\s:]*\$?(\d+\.\d{2})', line, re.IGNORECASE)
        if tax_match and not receipt.tax_amount:
            try:
                receipt.tax_amount = float(tax_match.group(1))
            except:
                pass

        # Payment method extraction
        payment_methods = ['CASH', 'CREDIT', 'DEBIT', 'CARD', 'VISA', 'MASTERCARD', 'AMEX']
        for method in payment_methods:
            if method in line.upper():
                receipt.payment_method = method.capitalize()
                break

    # Extract items (this is complex and varies greatly between receipts)
    # Basic implementation - looking for patterns like "ITEM NAME 1.00 2.00"
    item_pattern = r'([\w\s]+)\s+(\d+(?:\.\d{2})?)\s+(\d+(?:\.\d{2})?)'

    items_section = False
    for i, line in enumerate(lines):
        # Skip header lines (approximate)
        if i < 3:
            continue

        # Skip if we've reached the total section
        if "TOTAL" in line.upper() or "SUBTOTAL" in line.upper():
            break

        # Check for item pattern
        item_match = re.search(item_pattern, line)
        if item_match:
            try:
                description = item_match.group(1).strip()
                # Assuming the pattern captures quantity and price
                unit_price = float(item_match.group(2))
                total_price = float(item_match.group(3))
                quantity = 1.0  # Default if not explicitly stated

                # Calculate quantity if not provided
                if unit_price > 0:
                    quantity = round(total_price / unit_price, 2)

                receipt.items.append(ReceiptItem(description, quantity, unit_price, total_price))
            except:
                pass

    return receipt


def prepare_for_airtable(receipt_data):
    """
    Convert receipt data to a format ready for Airtable insertion
    """
    if not receipt_data:
        return None

    # Create main receipt record for Airtable
    receipt_record = {
        "fields": {
            "Store Name": receipt_data.store_name or "Unknown Store",
            "Date": receipt_data.date.strftime('%Y-%m-%d') if receipt_data.date else None,
            "Time": receipt_data.time,
            "Total Amount": receipt_data.total_amount,
            "Tax Amount": receipt_data.tax_amount,
            "Payment Method": receipt_data.payment_method,
            "Item Count": len(receipt_data.items)
        }
    }

    # Create item records for Airtable
    item_records = []
    for item in receipt_data.items:
        item_record = {
            "fields": {
                "Description": item.description,
                "Quantity": item.quantity,
                "Unit Price": item.unit_price,
                "Total Price": item.total_price,
                # Clé étrangère à remplir après création du reçu
                # "Receipt ID": [receipt_id]
            }
        }
        item_records.append(item_record)

    return {
        "receipt": receipt_record,
        "items": item_records
    }


def save_to_airtable(airtable_data):
    """
    Save receipt data to Airtable
    """
    if not AIRTABLE_API_KEY:
        raise ValueError("Airtable API key not configured")

    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }

    # Créer l'enregistrement du reçu
    receipt_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_RECEIPTS_TABLE}"
    receipt_response = requests.post(receipt_url, headers=headers, data=json.dumps({
        "records": [airtable_data["receipt"]]
    }))

    if receipt_response.status_code != 200:
        error_msg = f"Failed to create receipt record: {receipt_response.text}"
        print(error_msg)
        return {"error": error_msg}

    receipt_record = receipt_response.json()["records"][0]
    receipt_id = receipt_record["id"]

    # Ajouter l'ID du reçu à chaque article
    items_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_ITEMS_TABLE}"
    for item in airtable_data["items"]:
        item["fields"]["Receipt"] = [receipt_id]

    # Créer les enregistrements d'articles par lots (max 10 par requête)
    batch_size = 10
    items_responses = []

    for i in range(0, len(airtable_data["items"]), batch_size):
        batch = airtable_data["items"][i:i + batch_size]
        items_response = requests.post(items_url, headers=headers, data=json.dumps({
            "records": batch
        }))

        if items_response.status_code != 200:
            error_msg = f"Failed to create item records: {items_response.text}"
            print(error_msg)
            return {"error": error_msg, "receipt_id": receipt_id}

        items_responses.append(items_response.json())

    return {
        "success": True,
        "receipt_id": receipt_id,
        "receipt": receipt_record,
        "items_batches": items_responses
    }


@app.route('/scan-receipt', methods=['POST'])
def scan_receipt_endpoint():
    """
    API endpoint to scan receipt and extract data
    """
    # Vérifier si une image est présente dans la requête
    if 'image' not in request.files and 'image' not in request.form:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_data = None

        # Traitement selon le type de données envoyées
        if 'image' in request.files:
            # Image file upload
            file = request.files['image']
            image_bytes = file.read()
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image_data = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif 'image' in request.form:
            # Base64 encoded image
            base64_image = request.form['image']
            # Supprimer le préfixe si présent (ex: "data:image/jpeg;base64,")
            if ',' in base64_image:
                base64_image = base64_image.split(',')[1]

            image_bytes = base64.b64decode(base64_image)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image_data = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_data is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Prétraitement de l'image
        processed_img = preprocess_image(image_data)
        

        # Extraction du texte
        text = extract_text(processed_img)

        # Analyse du texte pour extraire les données structurées
        receipt_data = parse_receipt(text)

        # Préparation pour Airtable
        airtable_data = prepare_for_airtable(receipt_data)

        # Enregistrer dans Airtable si l'option est activée
        airtable_result = None
        save_to_db = request.form.get('save_to_db', 'true').lower() == 'true'

        if save_to_db and AIRTABLE_API_KEY:
            try:
                airtable_result = save_to_airtable(airtable_data)
            except Exception as e:
                airtable_result = {"error": str(e)}

        # Ajouter le texte brut pour le débogage
        response_data = {
            'raw_text': text,
            'receipt_data': {
                'store_name': receipt_data.store_name,
                'date': receipt_data.date.strftime('%Y-%m-%d') if receipt_data.date else None,
                'time': receipt_data.time,
                'total_amount': receipt_data.total_amount,
                'tax_amount': receipt_data.tax_amount,
                'payment_method': receipt_data.payment_method,
                'items': [
                    {
                        'description': item.description,
                        'quantity': item.quantity,
                        'unit_price': item.unit_price,
                        'total_price': item.total_price
                    } for item in receipt_data.items
                ]
            }
        }

        if airtable_result:
            response_data['airtable_result'] = airtable_result

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint
    """
    airtable_status = "configured" if AIRTABLE_API_KEY else "not configured"
    return jsonify({
        'status': 'ok',
        'service': 'receipt-scanner',
        'airtable': airtable_status
    }), 200


# Configuration pour le développement
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)