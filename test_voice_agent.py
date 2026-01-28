#!/usr/bin/env python3
"""
Простой скрипт для тестирования голосового агента
Можно использовать для быстрого тестирования без голосового ввода
"""

from voice_agent import VoiceAgent, Colors, print_colored
from personal_agent import check_ollama_available, get_available_models


def test_with_text_input(model: str = 'qwen2.5:7b'):
    """
    Тестирует агента с текстовым вводом (имитация распознанной речи)
    Полезно для быстрого тестирования без микрофона
    """
    test_queries = [
        "посчитай 25 умножить на 17",
        "дай определение искусственного интеллекта",
        "скажи анекдот"
    ]
    
    print_colored("\n" + "="*50, Colors.MAGENTA)
    print_colored("  🧪 ТЕСТИРОВАНИЕ ГОЛОСОВОГО АГЕНТА (ТЕКСТОВЫЙ РЕЖИМ)", Colors.BOLD)
    print_colored("="*50, Colors.MAGENTA)
    print_colored(f"  Модель: {model}", Colors.CYAN)
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
        print_colored(f"Доступные модели: {', '.join(models)}", Colors.YELLOW)
        return
    
    agent = VoiceAgent(model)
    
    print_colored("💡 Тестирование с текстовым вводом (имитация распознанной речи)\n", Colors.YELLOW)
    
    for i, query in enumerate(test_queries, 1):
        print_colored(f"\n{'='*50}", Colors.MAGENTA)
        print_colored(f"Тест {i}/{len(test_queries)}", Colors.BOLD)
        print_colored(f"{'='*50}", Colors.MAGENTA)
        
        print_colored(f"\n📝 Распознано (имитация): {query}", Colors.CYAN)
        print_colored("\n🤖 Ответ агента:\n", Colors.BOLD)
        
        try:
            response = agent.agent.process_message(query)
            print()
        except Exception as e:
            print_colored(f"\n❌ Ошибка: {e}", Colors.RED)
    
    print_colored("\n" + "="*50, Colors.MAGENTA)
    print_colored("✅ Тестирование завершено!", Colors.GREEN)
    print_colored("="*50 + "\n", Colors.MAGENTA)


if __name__ == '__main__':
    import sys
    
    model = 'qwen2.5:7b'
    if len(sys.argv) > 1:
        model = sys.argv[1]
    
    test_with_text_input(model)
