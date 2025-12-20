import os
import re
from typing import List, Dict

from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from google.oauth2.service_account import Credentials
import gspread
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE', 'service_account.json')
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID', '')
SHEET_NAME = os.getenv('SHEET_NAME', 'Sheet1')
SECRET_TOKEN = os.getenv('SECRET_TOKEN', 'please-change-me')

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly',
          'https://www.googleapis.com/auth/drive.readonly']

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET', 'change_this_secret')
socketio = SocketIO(app, cors_allowed_origins="*")

DATA_CACHE: List[Dict] = []


# -----------------------
# Normalisasi & utilitas
# -----------------------
def _clean_int(value):
    """Bersihkan string numerik: hapus non-digit lalu cast ke int, fallback 0."""
    if value is None or (isinstance(value, str) and value.strip() == ''):
        return 0
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except:
            return 0
    s = str(value)
    # hilangkan semua kecuali digit dan minus
    digits = re.sub(r'[^\d\-]', '', s)
    try:
        return int(digits) if digits not in ('', '-', None) else 0
    except:
        try:
            return int(float(digits))
        except:
            return 0


def _choose(keys, record: Dict):
    """
    Pilih value dari record berdasarkan daftar kemungkinan nama kolom.
    Mencoba exact key, lalu case-insensitive match terhadap keys di record.
    """
    for k in keys:
        if k in record and record[k] not in (None, ''):
            return record[k]
    # case-insensitive map
    lower_map = {rk.lower().strip(): rv for rk, rv in record.items()}
    for k in keys:
        lk = k.lower().strip()
        if lk in lower_map and lower_map[lk] not in (None, ''):
            return lower_map[lk]
    return None


def normalize_records(records: List[Dict]) -> List[Dict]:
    """
    Normalisasi record dari Google Sheets menjadi struktur yang dipakai dashboard.
    Mengembalikan list of dict dengan key:
    date, title, price, total, qty, status, buyer_email, buyer_name, buyer_phone,
    shipping_method, shipping_address, addon_detail, voucher_code, notes
    """
    out = []
    for r in records:
        # ambil kolom potensial (bahasa/variasi)
        title = _choose(['Judul Barang', 'Judul', 'title', 'Title', 'judul'], r) or ''
        price_raw = _choose(['Harga', 'harga', 'Price', 'price'], r)
        qty_raw = _choose(['qty', 'Qty', 'quantity', 'Quantity'], r)
        total_raw = _choose(['Total', 'total', 'sub total', 'Sub Total', 'sub_total', 'Subtotal', 'subtotal'], r)
        tanggal_raw = _choose(['Tanggal', 'tanggal', 'Date', 'date', 'Tanggal Transaksi', 'Transaction Date'], r)
        status = _choose(['Status', 'status', 'STATUS'], r) or ''
        buyer_email = _choose(['Buyer Email', 'buyer email', 'BuyerEmail', 'buyer_email', 'Email Pembeli', 'Email'], r) or ''
        buyer_name = _choose(['Buyer Name (opsional)', 'Buyer Name', 'buyer name', 'BuyerName', 'Buyer'], r) or ''
        buyer_phone = _choose(['Buyer Phone (Opsional)', 'Buyer Phone', 'buyer phone', 'Phone', 'No HP', 'BuyerPhone'], r) or ''
        shipping_method = _choose(['Shipping Method', 'shipping method', 'Shipping', 'Metode Pengiriman'], r) or ''
        shipping_address = _choose(['Shipping Address', 'shipping address', 'alamat pengiriman', 'Address'], r) or ''
        addon_detail = _choose(['Addon Detail', 'Addon', 'addon detail', 'Addon Detail'], r) or ''
        voucher_code = _choose(['Voucher Code', 'Voucher', 'voucher code', 'voucher'], r) or ''
        notes = _choose(['Notes (Opsional)', 'Notes', 'notes'], r) or ''

        # konversi numerik
        price_val = _clean_int(price_raw)
        total_val = _clean_int(total_raw)
        try:
            qty_val = int(qty_raw) if qty_raw not in (None, '') else 0
        except:
            qty_val = _clean_int(qty_raw)

        # parse tanggal ke ISO YYYY-MM-DD (gunakan pandas to_datetime untuk fleksibilitas)
        date_str = ''
        try:
            if tanggal_raw in (None, ''):
                date_str = ''
            else:
                dt = pd.to_datetime(tanggal_raw, dayfirst=True, errors='coerce')
                if pd.isna(dt):
                    dt = pd.to_datetime(tanggal_raw, dayfirst=False, errors='coerce')
                if pd.isna(dt):
                    date_str = str(tanggal_raw)
                else:
                    date_str = dt.strftime('%Y-%m-%d')
        except Exception:
            date_str = str(tanggal_raw)

        # jika price per unit kosong tapi total ada dan qty ada, hitung perkiraan price
        if price_val == 0 and total_val and qty_val:
            try:
                price_val = int(total_val / max(1, qty_val))
            except:
                pass

        out.append({
            'date': date_str,
            'title': str(title),
            'price': int(price_val),
            'total': int(total_val),
            'qty': int(qty_val),
            'status': str(status),
            'buyer_email': str(buyer_email),
            'buyer_name': str(buyer_name),
            'buyer_phone': str(buyer_phone),
            'shipping_method': str(shipping_method),
            'shipping_address': str(shipping_address),
            'addon_detail': str(addon_detail),
            'voucher_code': str(voucher_code),
            'notes': str(notes)
        })
    return out


# -----------------------
# Google Sheets loader
# -----------------------
def load_sheet_records() -> List[Dict]:
    """
    Muat seluruh sheet, normalisasi, dan return list of dict siap dipakai UI.
    """
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(f"Service account file tidak ditemukan: {SERVICE_ACCOUNT_FILE}")

    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(SPREADSHEET_ID)
    try:
        worksheet = sh.worksheet(SHEET_NAME)
    except Exception:
        worksheet = sh.get_worksheet(0)

    records = worksheet.get_all_records()  # list of dict (header -> value)
    normalized = normalize_records(records)
    return normalized


# -----------------------
# Routes & Webhook
# -----------------------
@app.route('/')
def index():
    return send_from_directory('.', 'dashboard.html')


@app.route('/data')
def data():
    global DATA_CACHE
    if not DATA_CACHE:
        try:
            DATA_CACHE = load_sheet_records()
        except Exception as e:
            app.logger.exception("Failed load sheet in /data: %s", e)
            return jsonify({'error': str(e)}), 500
    return jsonify(DATA_CACHE)


@app.route('/sheet-webhook', methods=['POST'])
def sheet_webhook():
    try:
        token = request.headers.get('X-SHEET-TOKEN', '')
        app.logger.info("Incoming webhook, headers: %s", dict(request.headers))
        # token check
        if token != SECRET_TOKEN:
            app.logger.warning("Invalid token: %s", token)
            return jsonify({'error': 'invalid token'}), 401

        payload = request.get_json(silent=True)
        app.logger.info('Payload received: %s', payload)

        # refresh cache — bungkus try/except untuk debugging
        try:
            global DATA_CACHE
            DATA_CACHE = load_sheet_records()
            app.logger.info("Loaded sheet records, rows=%d", len(DATA_CACHE))
        except Exception as e:
            tb = traceback.format_exc()
            app.logger.error("Error loading sheet: %s\n%s", e, tb)
            # kembalikan detail error untuk membantu debugging (development only)
            return jsonify({'error': 'failed to load sheet', 'detail': str(e), 'trace': tb}), 500

        # broadcast
        socketio.emit('sheet_update', {'payload': payload, 'count': len(DATA_CACHE)})
        return jsonify({'status': 'ok', 'rows': len(DATA_CACHE)}), 200

    except Exception as ex:
        tb = traceback.format_exc()
        app.logger.exception("Unhandled exception in sheet_webhook: %s", ex)
        return jsonify({'error': 'internal', 'detail': str(ex), 'trace': tb}), 500



@socketio.on('connect')
def handle_connect():
    app.logger.info('client connected')


# -----------------------
# Run
# -----------------------
if __name__ == '__main__':
    # jalankan dengan unbuffered output agar log muncul langsung:
    # python -u app.py
    socketio.run(app, host='0.0.0.0', port=5000)
