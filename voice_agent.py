#!/usr/bin/env python3
"""
Голосовой AI-агент
Ввод: голосовая команда → распознаётся в текст → отправляется в модель → ответ возвращается текстом
"""

import sys
import argparse
from personal_agent import PersonalAgent, check_ollama_available, get_available_models, Colors, print_colored
from voice_recognition import VoiceRecognizer


class VoiceAgent:
    """Голосовой агент с интеграцией распознавания речи и LLM"""
    
    def __init__(self, model: str, language: str = 'ru-RU'):
        """
        Инициализация голосового агента
        
        Args:
            model: Название модели Ollama
            language: Язык распознавания речи
        """
        self.agent = PersonalAgent(model)
        self.voice_recognizer = VoiceRecognizer(language=language)
        
    def process_voice_command(self, timeout: float = 5.0, phrase_time_limit: float = 10.0) -> str:
        """
        Обрабатывает голосовую команду
        
        Args:
            timeout: Максимальное время ожидания начала речи
            phrase_time_limit: Максимальная длительность фразы
            
        Returns:
            Ответ агента в виде текста
        """
        # 1. Распознаем речь
        recognized_text = self.voice_recognizer.recognize_from_microphone(
            timeout=timeout, 
            phrase_time_limit=phrase_time_limit
        )
        
        if not recognized_text:
            return None
        
        # 2. Показываем распознанный текст
        print_colored(f"\n📝 Распознано: {recognized_text}", Colors.CYAN)
        
        # 3. Отправляем в LLM
        print_colored("\n🤖 Ответ агента:\n", Colors.BOLD)
        response = self.agent.process_message(recognized_text)
        
        return response
    
    def interactive_voice_mode(self):
        """Интерактивный голосовой режим"""
        print_colored("\n" + "="*50, Colors.MAGENTA)
        print_colored("  🎤 ГОЛОСОВОЙ AI-АГЕНТ", Colors.BOLD)
        print_colored("="*50, Colors.MAGENTA)
        print_colored(f"  Модель: {self.agent.model}", Colors.CYAN)
        
        # Показываем информацию о памяти
        facts_count = len(self.agent.memory.memory['facts'])
        if facts_count > 0:
            print_colored(f"  Фактов в памяти: {facts_count}", Colors.GREEN)
            profile = self.agent.memory.memory['user_profile']
            if profile['name']:
                print_colored(f"  Пользователь: {profile['name']}", Colors.GREEN)
        
        print_colored("\n💡 Инструкции:", Colors.YELLOW)
        print_colored("  - Говорите в микрофон для отправки команды", Colors.CYAN)
        print_colored("  - Нажмите Ctrl+C для выхода", Colors.CYAN)
        print_colored("  - После распознавания команда будет отправлена в LLM", Colors.CYAN)
        print_colored("="*50 + "\n", Colors.MAGENTA)
        
        while True:
            try:
                print_colored("\n🎤 Говорите... (или Ctrl+C для выхода)", Colors.GREEN)
                
                response = self.process_voice_command()
                
                if response:
                    print()  # Пустая строка после ответа
                else:
                    print_colored("⚠️  Попробуйте еще раз\n", Colors.YELLOW)
                    
            except KeyboardInterrupt:
                print_colored("\n\n👋 До встречи!", Colors.YELLOW)
                break
            except Exception as e:
                print_colored(f"\n❌ Ошибка: {e}", Colors.RED)
                print_colored("Попробуйте еще раз\n", Colors.YELLOW)


def test_voice_agent(model: str, test_queries: list = None):
    """
    Тестирует голосового агента на нескольких запросах
    
    Args:
        model: Название модели Ollama
        test_queries: Список тестовых запросов (если None, использует стандартные)
    """
    if test_queries is None:
        test_queries = [
            "посчитай 25 умножить на 17",
            "дай определение искусственного интеллекта",
            "скажи анекдот"
        ]
    
    print_colored("\n" + "="*50, Colors.MAGENTA)
    print_colored("  🧪 ТЕСТИРОВАНИЕ ГОЛОСОВОГО АГЕНТА", Colors.BOLD)
    print_colored("="*50, Colors.MAGENTA)
    print_colored(f"  Модель: {model}", Colors.CYAN)
    print_colored(f"  Тестовых запросов: {len(test_queries)}", Colors.CYAN)
    print_colored("="*50 + "\n", Colors.MAGENTA)
    
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
    
    agent = VoiceAgent(model)
    
    print_colored("💡 ВАЖНО: Для тестирования вы можете:", Colors.YELLOW)
    print_colored("  1. Говорить запросы в микрофон (рекомендуется)", Colors.CYAN)
    print_colored("  2. Или ввести текст вручную для быстрого теста", Colors.CYAN)
    print_colored("\nВыберите режим:", Colors.YELLOW)
    print_colored("  [1] Голосовой режим (говорить в микрофон)", Colors.CYAN)
    print_colored("  [2] Текстовый режим (ввод с клавиатуры)", Colors.CYAN)
    
    try:
        choice = input("\nВаш выбор (1 или 2): ").strip()
        
        if choice == '2':
            # Текстовый режим для быстрого тестирования
            print_colored("\n📝 ТЕКСТОВЫЙ РЕЖИМ ТЕСТИРОВАНИЯ\n", Colors.BOLD)
            for i, query in enumerate(test_queries, 1):
                print_colored(f"\n{'='*50}", Colors.MAGENTA)
                print_colored(f"Тест {i}/{len(test_queries)}: {query}", Colors.BOLD)
                print_colored(f"{'='*50}", Colors.MAGENTA)
                
                print_colored(f"\n📝 Запрос: {query}", Colors.CYAN)
                print_colored("\n🤖 Ответ агента:\n", Colors.BOLD)
                
                response = agent.agent.process_message(query)
                print()
        else:
            # Голосовой режим
            print_colored("\n🎤 ГОЛОСОВОЙ РЕЖИМ ТЕСТИРОВАНИЯ\n", Colors.BOLD)
            print_colored("Примеры запросов для тестирования:", Colors.YELLOW)
            for i, query in enumerate(test_queries, 1):
                print_colored(f"  {i}. {query}", Colors.CYAN)
            
            print_colored("\nГоворите запросы в микрофон...\n", Colors.GREEN)
            agent.interactive_voice_mode()
            
    except KeyboardInterrupt:
        print_colored("\n\n👋 Тестирование прервано", Colors.YELLOW)
    except Exception as e:
        print_colored(f"\n❌ Ошибка: {e}", Colors.RED)


def main():
    parser = argparse.ArgumentParser(
        description='Голосовой AI-агент с распознаванием речи',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python voice_agent.py                    # Интерактивный голосовой режим
  python voice_agent.py -m qwen2.5:7b      # С указанием модели
  python voice_agent.py --test              # Тестирование на стандартных запросах
  python voice_agent.py --test --queries "посчитай 2+2" "дай определение"
        """
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='qwen2.5:7b',
        help='Модель Ollama (по умолчанию: qwen2.5:7b)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Запустить тестирование на стандартных запросах'
    )
    
    parser.add_argument(
        '--queries',
        nargs='+',
        help='Список тестовых запросов (используется с --test)'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default='ru-RU',
        help='Язык распознавания речи (по умолчанию: ru-RU)'
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_queries = args.queries if args.queries else None
        test_voice_agent(args.model, test_queries)
    else:
        # Интерактивный режим
        if not check_ollama_available():
            print_colored("❌ OLLama сервер недоступен. Запустите: ollama serve", Colors.RED)
            return
        
        models = get_available_models()
        if not models:
            print_colored("❌ Нет доступных моделей", Colors.RED)
            return
        
        if args.model not in models:
            print_colored(f"❌ Модель '{args.model}' не найдена", Colors.RED)
            return
        
        agent = VoiceAgent(args.model, language=args.language)
        agent.interactive_voice_mode()


if __name__ == '__main__':
    main()
