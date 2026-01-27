#!/usr/bin/env python3
"""
Персональный AI-агент с долговременной памятью
Запоминает всё о пользователе и использует эту информацию в диалогах
"""

import os
import sys
import json
import requests
import argparse
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import hashlib
import re

# URL OLLama API
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://127.0.0.1:11434')
OLLAMA_API_BASE = f"{OLLAMA_API_URL}/api"

# Директория для хранения данных
DATA_DIR = Path.home() / '.personal_agent'
DATA_DIR.mkdir(exist_ok=True)

MEMORY_FILE = DATA_DIR / 'memory.json'
CONVERSATIONS_DIR = DATA_DIR / 'conversations'
CONVERSATIONS_DIR.mkdir(exist_ok=True)

# Параметры модели
MODEL_CONFIG = {
    'temperature': 0.8,
    'num_predict': 2048,
    'num_ctx': 8192,
    'top_p': 0.9,
    'top_k': 40,
}

# Цвета для терминала
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    MAGENTA = '\033[35m'


def print_colored(text: str, color: str = Colors.END, end: str = '\n', flush: bool = False):
    """Печатает цветной текст"""
    print(f"{color}{text}{Colors.END}", end=end, flush=flush)


# ============================================================================
# СИСТЕМА ПАМЯТИ
# ============================================================================

class MemorySystem:
    """Управляет долговременной памятью агента о пользователе"""

    def __init__(self):
        self.memory = self._load_memory()
        self.current_session = datetime.now().strftime('%Y-%m-%d')

    def _load_memory(self) -> Dict:
        """Загружает память из файла"""
        if MEMORY_FILE.exists():
            try:
                with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print_colored(f"⚠️  Ошибка загрузки памяти: {e}", Colors.YELLOW)
                return self._empty_memory()
        return self._empty_memory()

    def _empty_memory(self) -> Dict:
        """Создает пустую структуру памяти"""
        return {
            'user_profile': {
                'name': None,
                'nickname': None,
                'age': None,
                'location': None,
                'occupation': None,
                'interests': [],
                'goals': [],
                'preferences': {}
            },
            'facts': [],  # Факты, которые агент узнал
            'conversations_summary': [],  # Сводки предыдущих разговоров
            'important_dates': {},  # Важные даты (день рождения и т.д.)
            'relationships': {},  # Информация о близких людях
            'habits': [],  # Привычки
            'created_at': datetime.now().isoformat(),
            'updated_at': None
        }

    def _save_memory(self):
        """Сохраняет память в файл"""
        self.memory['updated_at'] = datetime.now().isoformat()
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def add_fact(self, fact: str, category: str = 'general'):
        """Добавляет новый факт о пользователе"""
        fact_entry = {
            'fact': fact,
            'category': category,
            'added_at': datetime.now().isoformat(),
            'session': self.current_session
        }
        self.memory['facts'].append(fact_entry)
        self._save_memory()
        return fact_entry

    def update_profile(self, field: str, value: Any):
        """Обновляет поле профиля пользователя"""
        if field in self.memory['user_profile']:
            self.memory['user_profile'][field] = value
            self._save_memory()
            return True
        return False

    def add_interest(self, interest: str):
        """Добавляет интерес пользователя"""
        if interest not in self.memory['user_profile']['interests']:
            self.memory['user_profile']['interests'].append(interest)
            self._save_memory()

    def add_goal(self, goal: str):
        """Добавляет цель пользователя"""
        if goal not in self.memory['user_profile']['goals']:
            self.memory['user_profile']['goals'].append(goal)
            self._save_memory()

    def add_preference(self, key: str, value: str):
        """Добавляет предпочтение пользователя"""
        self.memory['user_profile']['preferences'][key] = value
        self._save_memory()

    def add_relationship(self, name: str, relation: str, details: str = ''):
        """Добавляет информацию о близком человеке"""
        self.memory['relationships'][name] = {
            'relation': relation,
            'details': details,
            'added_at': datetime.now().isoformat()
        }
        self._save_memory()

    def add_important_date(self, name: str, date: str, description: str = ''):
        """Добавляет важную дату"""
        self.memory['important_dates'][name] = {
            'date': date,
            'description': description,
            'added_at': datetime.now().isoformat()
        }
        self._save_memory()

    def add_habit(self, habit: str):
        """Добавляет привычку пользователя"""
        if habit not in self.memory['habits']:
            self.memory['habits'].append(habit)
            self._save_memory()

    def get_memory_context(self) -> str:
        """Формирует контекст из памяти для промпта"""
        context_parts = []
        has_any_info = False

        # Профиль
        profile = self.memory['user_profile']
        if profile['name']:
            context_parts.append(f"Имя пользователя: {profile['name']}")
            has_any_info = True
        if profile['nickname']:
            context_parts.append(f"Прозвище: {profile['nickname']}")
            has_any_info = True
        if profile['age']:
            context_parts.append(f"Возраст: {profile['age']}")
            has_any_info = True
        if profile['location']:
            context_parts.append(f"Местоположение: {profile['location']}")
            has_any_info = True
        if profile['occupation']:
            context_parts.append(f"Род занятий: {profile['occupation']}")
            has_any_info = True

        if profile['interests']:
            context_parts.append(f"Интересы: {', '.join(profile['interests'])}")
            has_any_info = True

        if profile['goals']:
            context_parts.append(f"Цели: {', '.join(profile['goals'])}")
            has_any_info = True

        if profile['preferences']:
            prefs = [f"{k}: {v}" for k, v in profile['preferences'].items()]
            context_parts.append(f"Предпочтения: {', '.join(prefs)}")
            has_any_info = True

        # Близкие люди
        if self.memory['relationships']:
            rels = [f"{name} ({info['relation']})" for name, info in self.memory['relationships'].items()]
            context_parts.append(f"Близкие люди: {', '.join(rels)}")
            has_any_info = True

        # Важные даты
        if self.memory['important_dates']:
            dates = [f"{name}: {info['date']}" for name, info in self.memory['important_dates'].items()]
            context_parts.append(f"Важные даты: {', '.join(dates)}")
            has_any_info = True

        # Привычки
        if self.memory['habits']:
            context_parts.append(f"Привычки: {', '.join(self.memory['habits'])}")
            has_any_info = True

        # Последние факты
        if self.memory['facts']:
            recent_facts = self.memory['facts'][-10:]  # Последние 10 фактов
            facts_list = [f['fact'] for f in recent_facts]
            context_parts.append(f"Что я знаю о пользователе:\n" + "\n".join(f"- {f}" for f in facts_list))
            has_any_info = True

        if not has_any_info:
            return "ПАМЯТЬ ПУСТАЯ: Я ещё ничего не знаю о пользователе. Это первый разговор."
        
        return "\n\n".join(context_parts)

    def get_profile_summary(self) -> str:
        """Возвращает сводку профиля"""
        lines = ["👤 ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ", "=" * 40]

        profile = self.memory['user_profile']
        has_profile_data = False

        if profile['name']:
            lines.append(f"📛 Имя: {profile['name']}")
            has_profile_data = True
        if profile['nickname']:
            lines.append(f"🔸 Прозвище: {profile['nickname']}")
            has_profile_data = True
        if profile['age']:
            lines.append(f"🎂 Возраст: {profile['age']}")
            has_profile_data = True
        if profile['location']:
            lines.append(f"📍 Местоположение: {profile['location']}")
            has_profile_data = True
        if profile['occupation']:
            lines.append(f"💼 Род занятий: {profile['occupation']}")
            has_profile_data = True

        if profile['interests']:
            lines.append(f"\n❤️ Интересы:")
            for interest in profile['interests']:
                lines.append(f"   • {interest}")

        if profile['goals']:
            lines.append(f"\n🎯 Цели:")
            for goal in profile['goals']:
                lines.append(f"   • {goal}")

        if profile['preferences']:
            lines.append(f"\n⚙️ Предпочтения:")
            for key, value in profile['preferences'].items():
                lines.append(f"   • {key}: {value}")

        if self.memory['relationships']:
            lines.append(f"\n👨‍👩‍👧‍👦 Близкие люди:")
            for name, info in self.memory['relationships'].items():
                lines.append(f"   • {name}: {info['relation']}")
                if info['details']:
                    lines.append(f"     {info['details']}")

        if self.memory['important_dates']:
            lines.append(f"\n📅 Важные даты:")
            for name, info in self.memory['important_dates'].items():
                lines.append(f"   • {name}: {info['date']}")
                if info['description']:
                    lines.append(f"     {info['description']}")

        if self.memory['habits']:
            lines.append(f"\n🔄 Привычки:")
            for habit in self.memory['habits']:
                lines.append(f"   • {habit}")

        if self.memory['facts']:
            lines.append(f"\n📝 Все факты ({len(self.memory['facts'])}):")
            for fact in self.memory['facts'][-20:]:  # Последние 20
                lines.append(f"   • {fact['fact']}")
            has_profile_data = True

        # Если профиль пустой, показываем сообщение
        if not has_profile_data:
            lines.append("\nПрофиль пуст. Используйте команду /set для заполнения или просто расскажите о себе.")

        return "\n".join(lines)

    def search_facts(self, query: str) -> List[Dict]:
        """Ищет факты по ключевому слову"""
        query_lower = query.lower()
        results = []
        for fact in self.memory['facts']:
            if query_lower in fact['fact'].lower():
                results.append(fact)
        return results

    def export_memory(self, filepath: str):
        """Экспортирует память в файл"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def import_memory(self, filepath: str):
        """Импортирует память из файла"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.memory.update(data)
            self._save_memory()


class ConversationLogger:
    """Логирует conversation_history для анализа"""

    def __init__(self):
        self.current_file = None

    def start_session(self, memory: MemorySystem):
        """Начинает новую сессию логирования"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_file = CONVERSATIONS_DIR / f'conversation_{timestamp}.jsonl'

    def log_exchange(self, user_message: str, assistant_response: str,
                     extracted_facts: List[str] = None):
        """Логирует обмен сообщениями"""
        if not self.current_file:
            return

        entry = {
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'assistant': assistant_response,
            'extracted_facts': extracted_facts or []
        }

        with open(self.current_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# ============================================================================
# OLLAMA API
# ============================================================================

def check_ollama_available() -> bool:
    """Проверяет доступность OLLama сервера"""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_available_models() -> List[str]:
    """Получает список доступных моделей"""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return [model['name'] for model in data.get('models', [])]
    except Exception as e:
        print_colored(f"Ошибка при получении списка моделей: {e}", Colors.RED)
        return []


def chat_with_model(model: str, messages: List[Dict[str, str]],
                    system_prompt: str = None, stream: bool = True) -> str:
    """Отправляет сообщения в модель через chat API"""
    url = f"{OLLAMA_API_BASE}/chat"

    # Формируем сообщения с system prompt
    if system_prompt:
        messages_with_system = [{"role": "system", "content": system_prompt}] + messages
    else:
        messages_with_system = messages

    payload = {
        "model": model,
        "messages": messages_with_system,
        "stream": stream,
        "options": MODEL_CONFIG
    }

    try:
        response = requests.post(url, json=payload, stream=stream, timeout=300)
        response.raise_for_status()

        if stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'message' in data and 'content' in data['message']:
                            chunk = data['message']['content']
                            print(chunk, end='', flush=True)
                            full_response += chunk
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            print()
            return full_response
        else:
            data = response.json()
            return data.get('message', {}).get('content', '')
    except requests.exceptions.RequestException as e:
        print_colored(f"Ошибка при запросе к модели: {e}", Colors.RED)
        return ""


def extract_facts_from_conversation(model: str, user_message: str,
                                     memory_context: str) -> List[str]:
    """Извлекает новые факты из сообщения пользователя"""
    # Если сообщение короткое и выглядит как имя - извлекаем его напрямую
    words = user_message.strip().split()
    if len(words) <= 2:
        first_word = words[0] if words else ""
        # Проверяем, что это может быть имя (начинается с заглавной, только буквы)
        if (first_word and 
            first_word[0].isupper() and 
            first_word.isalpha() and 
            len(first_word) >= 2):
            stop_words = {'привет', 'пока', 'да', 'нет', 'спасибо', 'как', 'что', 'где', 'когда'}
            if first_word.lower() not in stop_words:
                return [f"Пользователя зовут {first_word}"]
    
    prompt = f"""Проанализируй сообщение пользователя и выдели новую информацию о нём.

КОНТЕКСТ (то, что я уже знаю):
{memory_context}

СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:
{user_message}

Верни ТОЛЬКО новые факты в формате JSON-списка строк:
[
  "факт 1",
  "факт 2"
]

Если новой информации нет - верни пустой список [].

Важно:
- Извлекай конкретные факты (имена, даты, предпочтения, интересы)
- Если пользователь просто назвал имя (например, "Ололо") - извлеки факт "Пользователя зовут Ололо"
- Не извлекай временные состояния ("мне грустно")
- Факты должны быть постоянными ("любит джаз", а не "сейчас слушает джаз")
- ВСЕ факты должны быть написаны ТОЛЬКО на русском языке
"""

    response = chat_with_model(
        model,
        [{"role": "user", "content": prompt}],
        stream=False
    )

    try:
        # Извлекаем JSON из ответа
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            facts = json.loads(json_str)
            return [f for f in facts if f and isinstance(f, str)]
    except:
        pass

    return []


def update_profile_from_facts(model: str, facts: List[str], memory: MemorySystem) -> bool:
    """Анализирует факты и обновляет профиль пользователя"""
    if not facts:
        return False
    
    # Формируем список фактов для анализа
    facts_text = "\n".join([f"- {fact}" for fact in facts])
    current_profile = memory.memory['user_profile']
    
    prompt = f"""Проанализируй следующие факты о пользователе и извлеки структурированную информацию для профиля.

ФАКТЫ:
{facts_text}

ТЕКУЩИЙ ПРОФИЛЬ:
- Имя: {current_profile.get('name') or 'не указано'}
- Прозвище: {current_profile.get('nickname') or 'не указано'}
- Возраст: {current_profile.get('age') or 'не указано'}
- Местоположение: {current_profile.get('location') or 'не указано'}
- Род занятий: {current_profile.get('occupation') or 'не указано'}

Верни ТОЛЬКО JSON объект с полями профиля, которые нужно обновить (только те, которые найдены в фактах):
{{
  "name": "полное имя или null",
  "nickname": "прозвище или null",
  "age": число или null,
  "location": "город/место или null",
  "occupation": "род занятий или null"
}}

Если информации для поля нет - верни null для этого поля.
Если поле уже заполнено и новая информация не противоречит - можно оставить null.
ВАЖНО: age должно быть ЧИСЛОМ (не строкой), например 25, а не "25".

Примеры:
- Если факт: "меня зовут Ололол, но ты можешь звать меня Ололоша" → {{"name": "Ололол", "nickname": "Ололоша", "age": null, "location": null, "occupation": null}}
- Если факт: "мне 25 лет" → {{"name": null, "nickname": null, "age": 25, "location": null, "occupation": null}}
- Если факт: "я живу в Москве" → {{"name": null, "nickname": null, "age": null, "location": "Москва", "occupation": null}}
"""

    response = chat_with_model(
        model,
        [{"role": "user", "content": prompt}],
        stream=False
    )

    try:
        # Извлекаем JSON из ответа
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            profile_updates = json.loads(json_str)
            
            # Обновляем профиль
            updated = False
            for field, value in profile_updates.items():
                if value is not None and field in current_profile:
                    # Конвертируем возраст в число, если нужно
                    if field == 'age' and isinstance(value, str):
                        try:
                            value = int(value)
                        except ValueError:
                            continue  # Пропускаем невалидный возраст
                    
                    # Обновляем только если поле пустое или новое значение отличается
                    if current_profile[field] is None or current_profile[field] != value:
                        memory.update_profile(field, value)
                        updated = True
            
            return updated
    except Exception as e:
        # Если не удалось распарсить - не критично, просто не обновляем профиль
        pass

    return False


# ============================================================================
# ПЕРСОНАЛЬНЫЙ АГЕНТ
# ============================================================================

class PersonalAgent:
    """Персональный AI-агент с памятью"""

    def __init__(self, model: str):
        self.model = model
        self.memory = MemorySystem()
        self.logger = ConversationLogger()
        self.conversation_history = []
        self.logger.start_session(self.memory)

    def get_system_prompt(self) -> str:
        """Формирует персонализированный system prompt"""
        memory_context = self.memory.get_memory_context()
        has_memory = "ПАМЯТЬ ПУСТАЯ" not in memory_context

        return f"""ТЫ — ПЕРСОНАЛЬНЫЙ AI-АССИСТЕНТ, который знает пользователя лично и использует всю накопленную информацию.

⚠️ КРИТИЧЕСКИ ВАЖНО — ЯЗЫК ОБЩЕНИЯ:
- ОБЯЗАТЕЛЬНО отвечай ТОЛЬКО на русском языке
- ЗАПРЕЩЕНО использовать английский, китайский или любые другие языки
- ЗАПРЕЩЕНО смешивать языки в одном ответе
- Если нужно использовать иностранное слово — транслитерируй его на русский или объясни на русском
- Все твои ответы должны быть полностью на русском языке, без исключений

О ПОЛЬЗОВАТЕЛЕ (ТОЛЬКО ТО, ЧТО Я ДЕЙСТВИТЕЛЬНО ЗНАЮ):
{memory_context}

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
1. НИКОГДА не выдумывай информацию о пользователе, которой нет в разделе "О ПОЛЬЗОВАТЕЛЕ" выше
2. НИКОГДА не упоминай прошлые разговоры, если их нет в памяти
3. НИКОГДА не придумывай имена, события, планы или факты, которых нет в памяти
4. Если в памяти написано "ПАМЯТЬ ПУСТАЯ" - это первый разговор, не упоминай прошлые встречи
5. Если не знаешь что-то о пользователе - честно скажи "Я пока этого не знаю" или "Расскажи мне об этом"
6. Используй ТОЛЬКО информацию из раздела "О ПОЛЬЗОВАТЕЛЕ" выше - ничего другого

ТВОИ ПРИНЦИПЫ:
1. Используй информацию о пользователе для персонализации ответов (только если она есть в памяти)
2. Если пользователь рассказывает что-то новое о себе, это будет сохранено автоматически
3. Относись к пользователю дружелюбно и с пониманием
4. Давай рекомендации с учётом его интересов, целей и предпочтений (если они известны)
5. Будь кратким и по существу, но теплым и эмпатичным

ВАЖНО:
- Пользователь доверяет тебе личную информацию — относись к этому с уважением
- Если не знаешь ответа на основе имеющихся данных — честно скажи "Я пока этого не знаю"
- Узнавай новые детали о пользователе естественным образом в беседе
- {"Это не первый разговор - у тебя есть память о пользователе." if has_memory else "Это первый разговор - память пустая, узнавай о пользователе постепенно."}"""

    def process_message(self, user_message: str) -> str:
        """Обрабатывает сообщение пользователя"""
        # 1. Прямое извлечение имени (быстрый путь)
        name_extracted = self._extract_name_directly(user_message)
        
        # 2. Извлекаем новые факты
        memory_context = self.memory.get_memory_context()
        new_facts = extract_facts_from_conversation(
            self.model, user_message, memory_context
        )

        # 3. Если имя уже извлечено напрямую, добавляем факт для логирования
        if name_extracted:
            profile = self.memory.memory['user_profile']
            if profile.get('name'):
                # Проверяем, нет ли уже факта об имени
                has_name_fact = any('зовут' in fact.lower() or 'имя' in fact.lower() for fact in new_facts)
                if not has_name_fact:
                    new_facts.append(f"Пользователя зовут {profile['name']}")

        # 4. Сохраняем новые факты
        for fact in new_facts:
            self.memory.add_fact(fact)

        # 5. Обновляем профиль из фактов
        if new_facts:
            update_profile_from_facts(self.model, new_facts, self.memory)

        # 5. Добавляем сообщение в историю
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # 6. Получаем ответ от модели
        system_prompt = self.get_system_prompt()
        response = chat_with_model(
            self.model,
            self.conversation_history,
            system_prompt=system_prompt,
            stream=True
        )

        # 7. Логируем обмен
        self.logger.log_exchange(user_message, response, new_facts)

        # 8. Добавляем ответ в историю
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def _extract_name_directly(self, user_message: str):
        """Прямое извлечение имени из сообщения с помощью регулярных выражений"""
        message_lower = user_message.lower().strip()
        message_original = user_message.strip()
        profile = self.memory.memory['user_profile']
        
        # Список служебных слов, которые не являются именами
        stop_words = {'меня', 'зовут', 'мое', 'я', 'но', 'ты', 'можешь', 'звать', 'зови', 
                      'привет', 'пока', 'да', 'нет', 'спасибо', 'пожалуйста', 'как', 'что',
                      'где', 'когда', 'почему', 'кто', 'это', 'то', 'все', 'всё'}
        
        # Проверка: если сообщение короткое (1-2 слова) и выглядит как имя
        # Это работает только если профиль пустой (первое знакомство)
        words = message_original.split()
        if len(words) <= 2 and not profile.get('name') and not profile.get('nickname'):
            # Проверяем, начинается ли с заглавной буквы и состоит только из букв
            first_word = words[0]
            if (first_word and
                first_word[0].isupper() and 
                first_word.isalpha() and 
                len(first_word) >= 2 and
                first_word.lower() not in stop_words):
                # Это может быть имя - сохраняем его
                potential_name = first_word
                self.memory.update_profile('name', potential_name)
                return True
        
        # Паттерны для извлечения имени (более точные)
        name_patterns = [
            r'меня\s+зовут\s+([А-ЯЁа-яёA-Za-z]+)',
            r'мое\s+имя\s+([А-ЯЁа-яёA-Za-z]+)',
            r'я\s+—\s+([А-ЯЁа-яёA-Za-z]+)',
            r'я\s+([А-ЯЁа-яёA-Za-z]+)(?:\s*,|\s+но)',
        ]
        
        # Извлекаем имя
        name = None
        for pattern in name_patterns:
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                potential_name = match.group(1).strip()
                # Проверяем, что это не служебное слово
                if len(potential_name) > 2 and potential_name.lower() not in stop_words:
                    name = potential_name.capitalize()
                    break
        
        # Паттерны для прозвища (более точные)
        nickname_patterns = [
            r'но\s+ты\s+можешь\s+звать\s+меня\s+([А-ЯЁа-яёA-Za-z]+)',
            r'можешь\s+звать\s+меня\s+([А-ЯЁа-яёA-Za-z]+)',
            r'звать\s+меня\s+([А-ЯЁа-яёA-Za-z]+)',
            r'зови\s+меня\s+([А-ЯЁа-яёA-Za-z]+)',
            r'прозвище\s+([А-ЯЁа-яёA-Za-z]+)',
        ]
        
        nickname = None
        for pattern in nickname_patterns:
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                potential_nickname = match.group(1).strip()
                if len(potential_nickname) > 1 and potential_nickname.lower() not in stop_words:
                    nickname = potential_nickname.capitalize()
                    break
        
        # Обновляем профиль, если нашли имя
        name_found = False
        if name and (not profile.get('name') or profile.get('name') != name):
            self.memory.update_profile('name', name)
            name_found = True
        
        if nickname and (not profile.get('nickname') or profile.get('nickname') != nickname):
            self.memory.update_profile('nickname', nickname)
            name_found = True
        
        return name_found
    
    def clear_history(self):
        """Очищает историю текущего разговора (но не память!)"""
        self.conversation_history = []
        print_colored("💬 История текущего разговора очищена", Colors.GREEN)

    def show_memory(self):
        """Показывает всё, что агент знает о пользователе"""
        print_colored(self.memory.get_profile_summary(), Colors.CYAN)

    def search_memory(self, query: str):
        """Ищет факты в памяти"""
        results = self.memory.search_facts(query)
        if results:
            print_colored(f"\n🔍 Результаты поиска по '{query}':", Colors.BOLD)
            for fact in results:
                print_colored(f"  • {fact['fact']}", Colors.GREEN)
                print_colored(f"    Добавлен: {fact['added_at']}", Colors.END)
        else:
            print_colored(f"Ничего не найдено по запросу '{query}'", Colors.YELLOW)

    def add_fact_manual(self, fact: str, category: str = 'manual'):
        """Вручную добавляет факт"""
        self.memory.add_fact(fact, category)
        print_colored(f"✅ Факт сохранён: {fact}", Colors.GREEN)

    def set_profile_field(self, field: str, value: str):
        """Устанавливает поле профиля"""
        field_map = {
            'имя': 'name',
            'name': 'name',
            'ник': 'nickname',
            'nickname': 'nickname',
            'возраст': 'age',
            'age': 'age',
            'город': 'location',
            'location': 'location',
            'работа': 'occupation',
            'occupation': 'occupation',
        }

        field_key = field_map.get(field.lower())
        if field_key:
            self.memory.update_profile(field_key, value)
            print_colored(f"✅ {field} → {value}", Colors.GREEN)
        else:
            print_colored(f"❌ Неизвестное поле. Доступные: {', '.join(field_map.keys())}", Colors.RED)


# ============================================================================
# CLI
# ============================================================================

def interactive_mode(model: str):
    """Интерактивный режим персонального агента"""

    if not check_ollama_available():
        print_colored("❌ OLLama сервер недоступен. Запустите: ollama serve", Colors.RED)
        return

    models = get_available_models()
    if not models:
        print_colored("❌ Нет доступных моделей", Colors.RED)
        return

    if model not in models:
        print_colored(f"❌ Модель '{model}' не найдена", Colors.RED)
        return

    agent = PersonalAgent(model)

    # Приветствие
    print_colored("\n" + "="*50, Colors.MAGENTA)
    print_colored("  🧠 ПЕРСОНАЛЬНЫЙ AI-АГЕНТ", Colors.BOLD)
    print_colored("="*50, Colors.MAGENTA)
    print_colored(f"  Модель: {model}", Colors.CYAN)
    print_colored(f"  Память: {MEMORY_FILE}", Colors.CYAN)

    # Показываем сколько фактов уже известно
    facts_count = len(agent.memory.memory['facts'])
    if facts_count > 0:
        print_colored(f"  Фактов в памяти: {facts_count}", Colors.GREEN)
        profile = agent.memory.memory['user_profile']
        if profile['name']:
            print_colored(f"  Пользователь: {profile['name']}", Colors.GREEN)
    else:
        print_colored("  Новая память! Расскажите о себе.", Colors.YELLOW)

    print_colored("\nКоманды:", Colors.YELLOW)
    print_colored("  /help - справка", Colors.CYAN)
    print_colored("  /memory - показать всё, что я знаю", Colors.CYAN)
    print_colored("  /search <запрос> - поиск в памяти", Colors.CYAN)
    print_colored("  /fact <факт> - добавить факт вручную", Colors.CYAN)
    print_colored("  /set <поле> <значение> - установить поле профиля", Colors.CYAN)
    print_colored("  /clear - очистить историю разговора", Colors.CYAN)
    print_colored("  /exit - выход", Colors.CYAN)
    print_colored("="*50 + "\n", Colors.MAGENTA)

    while True:
        try:
            # Промпт
            profile = agent.memory.memory['user_profile']
            name = profile.get('name') or profile.get('nickname') or 'Друг'
            print_colored(f"[{name}] → ", Colors.GREEN, end='', flush=True)

            user_input = input().strip()

            if not user_input:
                continue

            # Команды
            if user_input.startswith('/'):
                parts = user_input.split(maxsplit=2)
                command = parts[0]

                if command in ['/exit', '/quit', '/q']:
                    print_colored("👋 До встречи! Я буду ждать тебя.", Colors.YELLOW)
                    break

                elif command == '/help':
                    print_colored("\n📖 СПРАВКА", Colors.BOLD)
                    print_colored("\nОбщение:", Colors.YELLOW)
                    print_colored("  Просто пишите сообщения, я запоминаю всё о вас!", Colors.CYAN)
                    print_colored("\nКоманды памяти:", Colors.YELLOW)
                    print_colored("  /memory - показать всё, что я знаю о вас", Colors.CYAN)
                    print_colored("  /search <запрос> - поиск в памяти", Colors.CYAN)
                    print_colored("  /fact <факт> - добавить факт вручную", Colors.CYAN)
                    print_colored("\nПрофиль:", Colors.YELLOW)
                    print_colored("  /set имя <ваше имя>", Colors.CYAN)
                    print_colored("  /set возраст <число>", Colors.CYAN)
                    print_colored("  /set город <город>", Colors.CYAN)
                    print_colored("  /set работа <род занятий>", Colors.CYAN)
                    print_colored("  /set ник <прозвище>", Colors.CYAN)
                    print_colored("\nУправление:", Colors.YELLOW)
                    print_colored("  /clear - очистить историю текущего разговора", Colors.CYAN)
                    print_colored("  /exit - выход", Colors.CYAN)
                    print()

                elif command == '/memory':
                    agent.show_memory()

                elif command == '/search':
                    if len(parts) < 2:
                        print_colored("❌ Укажите запрос: /search <запрос>", Colors.RED)
                    else:
                        agent.search_memory(parts[1])

                elif command == '/fact':
                    if len(parts) < 2:
                        print_colored("❌ Укажите факт: /fact <факт>", Colors.RED)
                    else:
                        agent.add_fact_manual(parts[1])

                elif command == '/set':
                    if len(parts) < 3:
                        print_colored("❌ Укажите поле и значение: /set <поле> <значение>", Colors.RED)
                        print_colored("   Поля: имя, ник, возраст, город, работа", Colors.YELLOW)
                    else:
                        agent.set_profile_field(parts[1], parts[2])

                elif command == '/clear':
                    agent.clear_history()

                else:
                    print_colored(f"❌ Неизвестная команда: {command}", Colors.RED)
                    print_colored("   Используйте /help для справки", Colors.YELLOW)

                continue

            # Обычное сообщение
            agent.process_message(user_input)
            print()

        except KeyboardInterrupt:
            print_colored("\n\n👋 До встречи!", Colors.YELLOW)
            break
        except EOFError:
            break
        except Exception as e:
            print_colored(f"\n❌ Ошибка: {e}", Colors.RED)


def main():
    parser = argparse.ArgumentParser(
        description='Персональный AI-агент с памятью',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python personal_agent.py
  python personal_agent.py -m qwen2.5:7b
  python personal_agent.py --show-memory
        """
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        default='qwen2.5:7b',
        help='Модель Ollama (по умолчанию: qwen2.5:7b)'
    )

    parser.add_argument(
        '--show-memory',
        action='store_true',
        help='Показать всю память и выйти'
    )

    parser.add_argument(
        '--export-memory',
        type=str,
        metavar='FILE',
        help='Экспортировать память в файл'
    )

    parser.add_argument(
        '--import-memory',
        type=str,
        metavar='FILE',
        help='Импортировать память из файла'
    )

    args = parser.parse_args()

    # Операции с памятью
    if args.show_memory:
        memory = MemorySystem()
        print(memory.get_profile_summary())
        return

    if args.export_memory:
        memory = MemorySystem()
        memory.export_memory(args.export_memory)
        print_colored(f"✅ Память экспортирована в {args.export_memory}", Colors.GREEN)
        return

    if args.import_memory:
        memory = MemorySystem()
        memory.import_memory(args.import_memory)
        print_colored(f"✅ Память импортирована из {args.import_memory}", Colors.GREEN)
        return

    # Интерактивный режим
    interactive_mode(args.model)


if __name__ == '__main__':
    main()
