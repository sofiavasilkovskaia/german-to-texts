import io
import streamlit as st
import requests
from PIL import Image
import time
import json
import os
from datetime import datetime
import base64
import pandas as pd
from typing import List, Dict, Tuple
import concurrent.futures
# import pdfkit
from fpdf import FPDF
import tempfile
from pathlib import Path
import hashlib
from functools import lru_cache
import shutil

# Константы
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
CACHE_DIR = Path('cache')
HISTORY_DIR = Path('history')
STATS_FILE = Path('stats.json')

# Создаем необходимые директории
CACHE_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)

# Настройка страницы
st.set_page_config(
    page_title="Распознавание текста",
    page_icon="📝",
    layout="wide"
)

# Инициализация состояния сессии
if 'history' not in st.session_state:
    st.session_state.history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []
if 'edited_texts' not in st.session_state:
    st.session_state.edited_texts = {}
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = {}
if 'export_files' not in st.session_state:
    st.session_state.export_files = {}
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_processed': 0,
        'total_success': 0,
        'total_failed': 0,
        'total_size': 0,
        'last_processed': None
    }

# Функции для работы с кэшем
@lru_cache(maxsize=100)
def get_cache_key(image_data: bytes) -> str:
    return hashlib.md5(image_data).hexdigest()

def save_to_cache(image_data: bytes, result: dict):
    cache_key = get_cache_key(image_data)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def get_from_cache(image_data: bytes) -> dict:
    cache_key = get_cache_key(image_data)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# Функции для работы со статистикой
def update_stats(success: bool, file_size: int):
    st.session_state.stats['total_processed'] += 1
    if success:
        st.session_state.stats['total_success'] += 1
    else:
        st.session_state.stats['total_failed'] += 1
    st.session_state.stats['total_size'] += file_size
    st.session_state.stats['last_processed'] = datetime.now().isoformat()
    
    # Сохраняем статистику
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.stats, f, ensure_ascii=False, indent=2)

# Функции для проверки безопасности
def is_allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_file_size(file_data: bytes) -> bool:
    return len(file_data) <= MAX_FILE_SIZE

def optimize_image(image_data: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(image_data))
        # Сжимаем изображение, если оно слишком большое
        if img.size[0] > 2000 or img.size[1] > 2000:
            img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
        # Конвертируем в JPEG для уменьшения размера
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=85)
        return output.getvalue()
    except Exception as e:
        st.error(f"Ошибка при оптимизации изображения: {str(e)}")
        return image_data

# Функции для пакетной обработки
def process_batch_images(images: List[Tuple[str, bytes]], settings: Dict) -> List[Dict]:
    results = []
    total = len(images)
    progress_text = "Обработка изображений..."
    progress_bar = st.progress(0, text=progress_text)
    
    # Создаем список для хранения результатов
    results = [None] * total
    
    # Обрабатываем изображения последовательно
    for i, (filename, image_data) in enumerate(images):
        try:
            result = process_single_image(image_data, settings)
            results[i] = result
            progress = (i + 1) / total
            progress_bar.progress(progress, text=f"{progress_text} {int(progress * 100)}%")
        except Exception as e:
            results[i] = {'error': str(e)}
            progress = (i + 1) / total
            progress_bar.progress(progress, text=f"{progress_text} {int(progress * 100)}%")
    
    progress_bar.empty()
    return results

def process_single_image(image_data: bytes, settings: Dict) -> Dict:
    try:
        # Проверяем кэш
        cached_result = get_from_cache(image_data)
        if cached_result:
            return cached_result
        
        # Оптимизируем изображение
        optimized_image = optimize_image(image_data)
        
        # Проверяем размер
        if not check_file_size(optimized_image):
            return {'error': 'Файл слишком большой'}
        
        files = {'image': optimized_image}
        response = requests.post('http://localhost:5000/recognize', files=files, data=settings)
        
        if response.status_code == 200:
            result = response.json()
            # Сохраняем в кэш
            save_to_cache(image_data, result)
            update_stats(True, len(image_data))
            return result
        else:
            update_stats(False, len(image_data))
            return {'error': f'Ошибка сервера: {response.json().get("error", "Неизвестная ошибка")}'}
    except Exception as e:
        update_stats(False, len(image_data))
        return {'error': str(e)}

# Функции для работы с историей
def save_to_history(image_data: bytes, text: str, language: str, processing_time: str):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Сохраняем изображение в отдельный файл
        image_hash = hashlib.md5(image_data).hexdigest()
        image_path = HISTORY_DIR / f"{image_hash}.jpg"
        with open(image_path, 'wb') as f:
            f.write(image_data)
        
        history_item = {
            'timestamp': timestamp,
            'text': text,
            'language': language,
            'processing_time': processing_time,
            'image_path': str(image_path)
        }
        
        st.session_state.history.append(history_item)
        history_file = HISTORY_DIR / 'history.json'
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Ошибка при сохранении истории: {str(e)}")

def delete_history_item(index: int):
    try:
        # Получаем реальный индекс в оригинальной истории
        original_index = len(st.session_state.history) - index - 1
        if 0 <= original_index < len(st.session_state.history):
            # Удаляем файл изображения
            item = st.session_state.history[original_index]
            image_path = Path(item['image_path'])
            if image_path.exists():
                image_path.unlink()
            
            # Удаляем запись
            st.session_state.history.pop(original_index)
            
            # Сохраняем обновленную историю
            history_file = HISTORY_DIR / 'history.json'
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
            st.success('Запись успешно удалена')
    except Exception as e:
        st.error(f'Ошибка при удалении записи: {str(e)}')

def clear_history():
    try:
        # Удаляем все файлы изображений
        for item in st.session_state.history:
            image_path = Path(item['image_path'])
            if image_path.exists():
                image_path.unlink()
        
        # Очищаем историю
        st.session_state.history = []
        history_file = HISTORY_DIR / 'history.json'
        if history_file.exists():
            history_file.unlink()
        
        st.success('История успешно очищена')
    except Exception as e:
        st.error(f'Ошибка при очистке истории: {str(e)}')

# Функции для экспорта
def export_to_txt(text: str, filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)

def export_to_pdf(text: str, filename: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(filename)

# Загрузка истории при старте
def load_history():
    try:
        history_file = HISTORY_DIR / 'history.json'
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                st.session_state.history = json.load(f)
    except Exception as e:
        st.warning(f"Не удалось загрузить историю: {str(e)}")
        st.session_state.history = []

# Загрузка статистики при старте
def load_stats():
    try:
        if STATS_FILE.exists():
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                st.session_state.stats = json.load(f)
    except Exception as e:
        st.warning(f"Не удалось загрузить статистику: {str(e)}")

# Загрузка данных при старте
load_history()
load_stats()

def get_supported_languages():
    try:
        response = requests.get('http://localhost:5000/languages')
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {
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

# Стилизация
st.markdown("""
    <style>
    .main {
        background-color: #000000;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .stTextInput>div>div>input {
        color: #ffffff;
        background-color: #1a1a1a;
        border: 1px solid #333;
    }
    .stSelectbox>div>div>select {
        color: #ffffff;
        background-color: #1a1a1a;
        border: 1px solid #333;
    }
    .stTextArea>div>div>textarea {
        color: #ffffff;
        background-color: #1a1a1a;
        border: 1px solid #333;
        font-family: monospace;
        font-size: 16px;
        line-height: 1.5;
    }
    .stMarkdown {
        color: #ffffff;
    }
    .stSubheader {
        color: #ffffff;
    }
    .stTitle {
        color: #ffffff;
    }
    .css-1d391kg {
        background-color: #1a1a1a;
    }
    .css-1y4p8pa {
        background-color: #1a1a1a;
    }
    .css-1v0mbdj {
        background-color: #1a1a1a;
    }
    .success {
        color: #4CAF50;
    }
    .error {
        color: #f44336;
    }
    .preview-image {
        max-width: 300px;
        max-height: 200px;
        object-fit: contain;
    }
    .history-item {
        background-color: #1a1a1a;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
        border: 1px solid #333;
    }
    .batch-result {
        background-color: #1a1a1a;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
        border: 1px solid #333;
    }
    .progress-bar {
        height: 4px;
        background-color: #4CAF50;
        transition: width 0.3s ease;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .pulse {
        animation: pulse 2s infinite;
    }
    </style>
""", unsafe_allow_html=True)

def compare_results(result1: Dict, result2: Dict) -> Dict:
    try:
        text1 = result1.get('text', '')
        text2 = result2.get('text', '')
        
        # Разбиваем тексты на слова
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Находим общие и различные слова
        common_words = words1.intersection(words2)
        unique_words1 = words1 - words2
        unique_words2 = words2 - words1
        
        return {
            'similarity': len(common_words) / max(len(words1), len(words2)) if words1 or words2 else 0,
            'common_words': list(common_words),
            'unique_words1': list(unique_words1),
            'unique_words2': list(unique_words2)
        }
    except Exception as e:
        return {'error': str(e)}

def load_image():
    uploaded_files = st.file_uploader(
        label='Выберите изображения для распознавания',
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help='Поддерживаются форматы JPG, JPEG и PNG'
    )
    if uploaded_files:
        images = []
        for file in uploaded_files:
            image_data = file.getvalue()
            st.image(image_data, caption=file.name, width=200)
            images.append((file, image_data))
        return images
    return None

# Настройки
with st.sidebar:
    st.header('⚙️ Настройки')
    
    # Качество распознавания
    quality = st.select_slider(
        'Качество распознавания',
        options=['Низкое', 'Среднее', 'Высокое'],
        value='Среднее'
    )
    
    # Дополнительные настройки
    with st.expander('Дополнительные настройки'):
        # Оптимизация изображений
        optimize = st.checkbox('Оптимизировать изображения', value=True)
        
        # Кэширование
        use_cache = st.checkbox('Использовать кэш', value=True)
        
        # Параллельная обработка
        parallel = st.checkbox('Параллельная обработка', value=True)
        
        # Максимальный размер файла
        max_size = st.number_input(
            'Максимальный размер файла (МБ)',
            min_value=1,
            max_value=50,
            value=10
        )
    
    # Статистика
    st.header('📊 Статистика')
    if st.session_state.stats['total_processed'] > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Всего обработано', st.session_state.stats['total_processed'])
            st.metric('Успешно', st.session_state.stats['total_success'])
        with col2:
            st.metric('Ошибок', st.session_state.stats['total_failed'])
            st.metric('Общий размер', f"{st.session_state.stats['total_size'] / 1024 / 1024:.1f} МБ")
        
        if st.session_state.stats['last_processed']:
            last_time = datetime.fromisoformat(st.session_state.stats['last_processed'])
            st.caption(f'Последняя обработка: {last_time.strftime("%Y-%m-%d %H:%M:%S")}')

# Основной контент
st.title('📝 Распознавание текста на изображениях')

# Загрузка изображений
uploaded_files = st.file_uploader(
    'Загрузите изображения',
    type=['png', 'jpg', 'jpeg', 'pdf'],
    accept_multiple_files=True
)

if uploaded_files:
    # Проверяем размер файлов
    total_size = sum(len(file.getvalue()) for file in uploaded_files)
    if total_size > max_size * 1024 * 1024:
        st.error(f'Общий размер файлов превышает {max_size} МБ')
    else:
        # Подготавливаем изображения
        images = []
        for file in uploaded_files:
            if is_allowed_file(file.name):
                image_data = file.getvalue()
                if check_file_size(image_data):
                    images.append((file.name, image_data))
                else:
                    st.warning(f'Файл {file.name} слишком большой и будет пропущен')
            else:
                st.warning(f'Неподдерживаемый формат файла: {file.name}')
        
        if images:
            # Настройки распознавания
            settings = {
                'quality': quality,
                'optimize': optimize,
                'use_cache': use_cache,
                'parallel': parallel
            }
            
            # Кнопка распознавания
            if st.button('Распознать текст'):
                st.session_state.processing = True
                try:
                    results = process_batch_images(images, settings)
                    
                    # Отображаем результаты
                    for i, (result, (filename, _)) in enumerate(zip(results, images)):
                        with st.expander(f'Результат для {filename}'):
                            if 'error' in result:
                                st.error(f'Ошибка: {result["error"]}')
                            else:
                                # Изображение
                                st.image(images[i][1], caption=filename, width=300)
                                
                                # Текст
                                text = result.get('text', '')
                                edited_text = st.text_area(
                                    'Распознанный текст',
                                    value=text,
                                    height=300,
                                    key=f'text_{i}'
                                )
                                
                                # Сохраняем в историю
                                if text != edited_text:
                                    st.session_state.edited_texts[i] = edited_text
                                save_to_history(
                                    images[i][1],
                                    edited_text,
                                    'ru',
                                    result.get('processing_time', '')
                                )
                except Exception as e:
                    st.error(f'Ошибка при обработке: {str(e)}')
                finally:
                    st.session_state.processing = False

# История
if st.session_state.history:
    st.header('📋 История распознавания')
    
    # Фильтры
    col1, col2 = st.columns(2)
    with col1:
        date_filter = st.date_input('Фильтр по дате')
    with col2:
        if st.button('Очистить историю'):
            clear_history()
            st.experimental_rerun()
    
    # Отображаем историю
    for i, item in enumerate(reversed(st.session_state.history)):
        item_date = datetime.strptime(item['timestamp'], "%Y-%m-%d %H:%M:%S").date()
        if date_filter is None or item_date == date_filter:
            with st.expander(f'Запись от {item["timestamp"]}'):
                col1, col2 = st.columns([3, 1])
                with col1:
                    image_path = Path(item['image_path'])
                    if image_path.exists():
                        st.image(str(image_path), width=300)
                    else:
                        st.warning('Изображение не найдено')
                with col2:
                    if st.button('🗑️ Удалить', key=f'delete_{i}'):
                        delete_history_item(i)
                        st.experimental_rerun()
                
                st.text_area('Текст', value=item['text'], height=300, key=f'history_text_{i}')
                st.caption(f'Время обработки: {item["processing_time"]}')

# Футер
st.markdown('---')
st.markdown('*Приложение для распознавания текста на изображениях*') 