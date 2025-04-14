from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging
import requests
import base64
import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Конфигурация OCR.space
OCR_SPACE_API_KEY = os.getenv('OCR_SPACE_API_KEY')
if not OCR_SPACE_API_KEY:
    logger.error("API ключ OCR.space не найден в переменных окружения!")
    raise ValueError("API ключ OCR.space не найден в переменных окружения!")
else:
    logger.info("API ключ OCR.space успешно загружен")

OCR_SPACE_URL = 'https://api.ocr.space/parse/image'

# Поддерживаемые языки
SUPPORTED_LANGUAGES = {
    'en': 'Английский',
    'ru': 'Русский',
    'de': 'Немецкий',
    'fr': 'Французский',
    'es': 'Испанский',
    'it': 'Итальянский',
    'pt': 'Португальский',
    'nl': 'Нидерландский',
    'pl': 'Польский',
    'uk': 'Украинский',
    'ja': 'Японский',
    'ko': 'Корейский',
    'zh': 'Китайский',
    'ar': 'Арабский'
}

# Соответствие языков для OCR.space
OCR_SPACE_LANGUAGES = {
    'en': 'eng',
    'ru': 'rus',
    'de': 'ger',
    'fr': 'fre',
    'es': 'spa',
    'it': 'ita',
    'pt': 'por',
    'nl': 'dut',
    'pl': 'pol',
    'uk': 'ukr',
    'ja': 'jpn',
    'ko': 'kor',
    'zh': 'chi_sim',
    'ar': 'ara'
}

def ocr_space_recognize(image_data, language='en'):
    try:
        # Определяем тип файла
        file_type = 'png'  # По умолчанию PNG
        if image_data.startswith(b'\xFF\xD8\xFF'):
            file_type = 'jpg'
        elif image_data.startswith(b'%PDF'):
            file_type = 'pdf'
            
        # Кодируем изображение в base64 с правильным префиксом
        base64_image = f"data:image/{file_type};base64,{base64.b64encode(image_data).decode('utf-8')}"
        
        # Преобразуем наш код языка в формат OCR.space
        ocr_space_lang = OCR_SPACE_LANGUAGES.get(language, 'eng')
        
        payload = {
            'base64Image': base64_image,
            'language': ocr_space_lang,
            'isOverlayRequired': False,
            'OCREngine': 2,  # 2 - лучший движок
            'filetype': file_type.upper(),
            'detectOrientation': True,
            'scale': True,
            'isCreateSearchablePdf': False,
            'isSearchablePdfHideTextLayer': False,
            'isTable': False
        }
        
        headers = {
            'apikey': OCR_SPACE_API_KEY,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        logger.info(f"Отправка запроса к OCR.space API с языком: {ocr_space_lang}")
        logger.info(f"Размер изображения: {len(image_data)} байт")
        logger.info(f"Тип файла: {file_type}")
        
        response = requests.post(OCR_SPACE_URL, data=payload, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Ошибка API OCR.space. Статус: {response.status_code}")
            logger.error(f"Ответ API: {response.text}")
            raise Exception(f"Ошибка API OCR.space: {response.text}")
            
        result = response.json()
        if result['IsErroredOnProcessing']:
            raise Exception(result['ErrorMessage'])
            
        text = result['ParsedResults'][0]['ParsedText']
        return text.strip()
        
    except Exception as e:
        logger.error(f"Ошибка при распознавании через OCR.space: {str(e)}")
        raise

@app.route('/languages', methods=['GET'])
def get_languages():
    return jsonify(SUPPORTED_LANGUAGES)

@app.route('/recognize', methods=['POST'])
def recognize_text():
    start_time = time.time()
    
    if 'image' not in request.files:
        return jsonify({'error': 'Изображение не предоставлено'}), 400
    
    try:
        lang = request.form.get('language', 'en')
        if lang not in SUPPORTED_LANGUAGES:
            return jsonify({'error': f'Неподдерживаемый язык: {lang}'}), 400
        
        file = request.files['image']
        image_data = file.read()
        
        text = ocr_space_recognize(image_data, lang)
        processing_time = time.time() - start_time
        
        return jsonify({
            'text': text,
            'processing_time': f'{processing_time:.2f} секунд',
            'language': SUPPORTED_LANGUAGES[lang],
            'detected_language': lang
        })
        
    except Exception as e:
        logger.error(f"Ошибка при распознавании текста: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)