#!/usr/bin/env python3
"""
Полный запуск проекта: Сравнение производительности оптимизации + профилирование FLOPS
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_header():
    """Выводит заголовок проекта"""
    print("=" * 80)
    print("ПОЛНЫЙ ЗАПУСК ПРОЕКТА: СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ ОПТИМИЗАЦИИ")
    print("=" * 80)
    print(f"Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_files():
    """Проверяет наличие всех необходимых файлов"""
    required_files = [
        'main.py',
        'train_models.py', 
        'benchmark.py',
        'generate_report.py',
        'flops_profiler.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Отсутствуют файлы:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print(" Все необходимые файлы найдены")
    return True

def run_step(step_name, script_name, description):
    """Запускает отдельный шаг проекта"""
    print(f"\n{'='*60}")
    print(f"ШАГ: {step_name}")
    print(f"{'='*60}")
    print(f"Описание: {description}")
    print()
    
    if not os.path.exists(script_name):
        print(f"Файл {script_name} не найден")
        return False
    
    try:
        print(f"Запуск {script_name}...")
        start_time = time.time()
        
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f" {step_name} завершен успешно за {duration:.1f} секунд")
            return True
        else:
            print(f" Ошибка в {step_name}:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Ошибка при запуске {script_name}: {e}")
        return False

def show_final_results():
    """Показывает финальные результаты"""
    print("\n" + "="*80)
    print(" ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
    print("="*80)
    
    # Проверяем наличие всех результатов
    result_files = [
        'benchmark_results.json',
        'benchmark_results.png', 
        'detailed_benchmark_analysis.png',
        'benchmark_report.txt',
        'flops_analysis.json',
        'flops_utilization_analysis.png'
    ]
    
    print("Созданные файлы результатов:")
    for file in result_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"{file} ({size:.1f} KB)")
        else:
            print(f" {file} - не найден")
    
    # Проверяем модели
    model_files = [
        'weights/best_resnet18_224.pth',
        'weights/best_resnet18_256.pth',
        'weights/best_resnet18_384.pth', 
        'weights/best_resnet18_512.pth'
    ]
    
    print("\nОбученные модели:")
    for model in model_files:
        if os.path.exists(model):
            size = os.path.getsize(model) / 1024**2  # MB
            print(f" {model} ({size:.1f} MB)")
        else:
            print(f" {model} - не найден")
    
    print("\n" + "="*80)
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*80)
    
    print("1. benchmark_results.json - детальные результаты бенчмарка")
    print("2. benchmark_results.png - основные графики производительности")
    print("3. detailed_benchmark_analysis.png - детальный анализ (6 графиков)")
    print("4. benchmark_report.txt - текстовый отчет")
    print("5. flops_analysis.json - анализ FLOPS и загруженности")
    print("6. flops_utilization_analysis.png - графики FLOPS и загруженности")
    
    print("\n📈 Ключевые метрики:")
    print("- FPS для каждого подхода оптимизации")
    print("- Ускорение относительно PyTorch")
    print("- Влияние размера изображения на производительность")
    print("- Оптимальный размер батча")
    print("- Загруженность GPU/CPU")
    print("- FLOPS анализ")
    
    print("\n🔍 Вопросы для анализа:")
    print("1. Какой подход показывает лучшую производительность?")
    print("2. Как размер изображения влияет на производительность?")
    print("3. Как размер батча влияет на производительность?")
    print("4. Какая загруженность железа достигается?")
    print("5. Какие оптимальные параметры для вашей системы?")

def main():
    """Главная функция"""
    print_header()
    
    # Проверка файлов
    if not check_files():
        print("Проект не может быть запущен из-за отсутствующих файлов")
        return
    
    # Шаг 1: Обучение моделей
    if not run_step(
        "ОБУЧЕНИЕ МОДЕЛЕЙ", 
        "train_models.py",
        "Обучение моделей ResNet-18 для размеров 224x224, 256x256, 384x384, 512x512"
    ):
        print("Обучение не завершено. Проверьте ошибки выше.")
        return
    
    # Шаг 2: Бенчмарк производительности
    if not run_step(
        "БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ",
        "benchmark.py", 
        "Сравнение производительности PyTorch, ONNX, TensorRT/ROCm"
    ):
        print("Бенчмарк не завершен. Проверьте ошибки выше.")
        return
    
    # Шаг 3: Генерация отчета
    if not run_step(
        "ГЕНЕРАЦИЯ ОТЧЕТА",
        "generate_report.py",
        "Создание графиков и текстового отчета"
    ):
        print("Отчет не сгенерирован. Проверьте ошибки выше.")
        return
    
    # Шаг 4: Профилирование FLOPS
    if not run_step(
        "ПРОФИЛИРОВАНИЕ FLOPS",
        "flops_profiler.py",
        "Анализ FLOPS и загруженности железа"
    ):
        print("Профилирование FLOPS не завершено. Проверьте ошибки выше.")
        return
    
    # Показ результатов
    show_final_results()
    
    print("\n" + "="*80)
    print(" ПРОЕКТ УСПЕШНО ЗАВЕРШЕН!")
    print("="*80)
    print("\n📋 Что было выполнено:")
    print("Обучены модели ResNet-18 для 4 размеров изображений")
    print("Проведен бенчмарк 3 подходов оптимизации")
    print("Созданы графики производительности")
    print("Выполнен анализ FLOPS и загруженности")
    print("Определены оптимальные параметры")
    

if __name__ == "__main__":
    main() 