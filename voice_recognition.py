#!/usr/bin/env python3
"""
Модуль распознавания речи
Использует speech_recognition для преобразования голоса в текст
"""

import speech_recognition as sr
import sys
from typing import Optional


class VoiceRecognizer:
    """Класс для распознавания речи"""
    
    def __init__(self, language: str = 'ru-RU'):
        """
        Инициализация распознавателя речи
        
        Args:
            language: Язык распознавания (по умолчанию ru-RU для русского)
        """
        self.recognizer = sr.Recognizer()
        self.language = language
        self.microphone = None
        
    def _get_microphone(self):
        """Получает микрофон (кэширует для повторного использования)"""
        if self.microphone is None:
            try:
                self.microphone = sr.Microphone()
            except Exception as e:
                raise RuntimeError(f"Не удалось получить доступ к микрофону: {e}")
        return self.microphone
    
    def recognize_from_microphone(self, timeout: float = 5.0, phrase_time_limit: float = 10.0) -> Optional[str]:
        """
        Распознает речь с микрофона
        
        Args:
            timeout: Максимальное время ожидания начала речи (секунды)
            phrase_time_limit: Максимальная длительность фразы (секунды)
            
        Returns:
            Распознанный текст или None в случае ошибки
        """
        microphone = self._get_microphone()
        
        # Настройка для подавления шума
        with microphone as source:
            print("🎤 Настраиваю микрофон... (пожалуйста, подождите)")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("✅ Готово! Говорите...")
            
        try:
            with microphone as source:
                # Записываем аудио
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
            
            print("🔍 Распознаю речь...")
            
            # Распознаем речь используя Google Speech Recognition
            try:
                text = self.recognizer.recognize_google(audio, language=self.language)
                return text
            except sr.UnknownValueError:
                print("❌ Не удалось распознать речь. Попробуйте еще раз.")
                return None
            except sr.RequestError as e:
                print(f"❌ Ошибка сервиса распознавания: {e}")
                print("💡 Убедитесь, что у вас есть интернет-соединение.")
                return None
                
        except sr.WaitTimeoutError:
            print("⏱️  Время ожидания истекло. Не услышал речи.")
            return None
        except Exception as e:
            print(f"❌ Ошибка при записи аудио: {e}")
            return None
    
    def recognize_from_file(self, audio_file: str) -> Optional[str]:
        """
        Распознает речь из аудио файла
        
        Args:
            audio_file: Путь к аудио файлу
            
        Returns:
            Распознанный текст или None в случае ошибки
        """
        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
            
            print("🔍 Распознаю речь из файла...")
            
            try:
                text = self.recognizer.recognize_google(audio, language=self.language)
                return text
            except sr.UnknownValueError:
                print("❌ Не удалось распознать речь в файле.")
                return None
            except sr.RequestError as e:
                print(f"❌ Ошибка сервиса распознавания: {e}")
                return None
                
        except FileNotFoundError:
            print(f"❌ Файл не найден: {audio_file}")
            return None
        except Exception as e:
            print(f"❌ Ошибка при обработке файла: {e}")
            return None


def test_voice_recognition():
    """Тестовая функция для проверки распознавания речи"""
    print("=" * 50)
    print("🎤 ТЕСТ РАСПОЗНАВАНИЯ РЕЧИ")
    print("=" * 50)
    
    try:
        recognizer = VoiceRecognizer()
        print("\nГоворите в микрофон...")
        text = recognizer.recognize_from_microphone()
        
        if text:
            print(f"\n✅ Распознанный текст: {text}")
        else:
            print("\n❌ Не удалось распознать речь")
            
    except KeyboardInterrupt:
        print("\n\n👋 Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == '__main__':
    test_voice_recognition()
