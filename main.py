#!/usr/bin/env python3
"""
Главный скрипт для выполнения домашнего задания по сравнению производительности
различных подходов оптимизации инференса нейронных сетей.

"""

import os
import sys
import argparse
import subprocess
import time

def check_dependencies():
    """Проверяет наличие необходимых зависимостей"""
    required_packages = [
        'torch', 'torchvision', 'onnx', 'onnxruntime', 
        'numpy', 'matplotlib', 'seaborn', 'pandas', 
        'tqdm', 'PIL', 'psutil', 'GPUtil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Отсутствуют необходимые пакеты:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nУстановите их командой:")
        print("pip install -r requirements.txt")
        return False
    
    print("Все зависимости установлены")
    return True

def check_gpu():
    """Проверяет доступность GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU обнаружен: {gpu_name}")
            print(f"Память: {gpu_memory:.1f} GB")
            return True
        else:
            print("GPU не обнаружен, будет использован CPU")
            return False
    except Exception as e:
        print(f"Ошибка при проверке GPU: {e}")
        return False

def run_training():
    """Запускает обучение моделей"""
    print("\n" + "="*60)
    print("ШАГ 1: ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("="*60)
    
    if not os.path.exists('train_models.py'):
        print("Файл train_models.py не найден")
        return False
    
    try:
        print("Запуск обучения моделей ResNet-18...")
        result = subprocess.run([sys.executable, 'train_models.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Обучение завершено успешно")
            return True
        else:
            print("Ошибка при обучении:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Ошибка при запуске обучения: {e}")
        return False

def run_benchmark():
    """Запускает бенчмарк производительности"""
    print("\n" + "="*60)
    print("ШАГ 2: БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*60)
    
    if not os.path.exists('benchmark.py'):
        print("Файл benchmark.py не найден")
        return False
    
    try:
        print("Запуск бенчмарка производительности...")
        result = subprocess.run([sys.executable, 'benchmark.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Бенчмарк завершен успешно")
            return True
        else:
            print("Ошибка при бенчмарке:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Ошибка при запуске бенчмарка: {e}")
        return False

def generate_report():
    """Генерирует отчет"""
    print("\n" + "="*60)
    print("ШАГ 3: ГЕНЕРАЦИЯ ОТЧЕТА")
    print("="*60)
    
    if not os.path.exists('generate_report.py'):
        print("Файл generate_report.py не найден")
        return False
    
    try:
        print("Генерация отчета...")
        result = subprocess.run([sys.executable, 'generate_report.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Отчет сгенерирован успешно")
            return True
        else:
            print("Ошибка при генерации отчета:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Ошибка при генерации отчета: {e}")
        return False

def show_results():
    """Показывает результаты"""
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ")
    print("="*60)
    
    # Проверяем наличие файлов результатов
    result_files = [
        'benchmark_results.json',
        'benchmark_results.png',
        'detailed_benchmark_analysis.png',
        'benchmark_report.txt'
    ]
    
    for file in result_files:
        if os.path.exists(file):
            print(f"{file}")
        else:
            print(f"{file} - не найден")
    
    # Проверяем наличие обученных моделей
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
            print(f"{model} ({size:.1f} MB)")
        else:
            print(f"{model} - не найден")

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Сравнение производительности оптимизации нейронных сетей')
    parser.add_argument('--skip-training', action='store_true', 
                       help='Пропустить обучение моделей')
    parser.add_argument('--skip-benchmark', action='store_true', 
                       help='Пропустить бенчмарк')
    parser.add_argument('--skip-report', action='store_true', 
                       help='Пропустить генерацию отчета')
    parser.add_argument('--only-report', action='store_true', 
                       help='Только генерация отчета')
    
    args = parser.parse_args()
    
    print("🚀 ЗАПУСК ПРОЕКТА: СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ ОПТИМИЗАЦИИ")
    print("="*80)
    
    # Проверка зависимостей
    if not check_dependencies():
        return
    
    # Проверка GPU
    check_gpu()
    
    # Если только отчет
    if args.only_report:
        generate_report()
        show_results()
        return
    
    # Обучение моделей
    if not args.skip_training:
        if not run_training():
            print("Обучение не завершено. Проверьте ошибки выше.")
            return
    
    # Бенчмарк производительности
    if not args.skip_benchmark:
        if not run_benchmark():
            print("Бенчмарк не завершен. Проверьте ошибки выше.")
            return
    
    # Генерация отчета
    if not args.skip_report:
        if not generate_report():
            print("Отчет не сгенерирован. Проверьте ошибки выше.")
            return
    
    # Показ результатов
    show_results()
    
    print("\n" + "="*80)
    print("ПРОЕКТ ЗАВЕРШЕН УСПЕШНО!")
    print("="*80)
    print("\nСозданные файлы:")
    print("- benchmark_results.json - результаты бенчмарка")
    print("- benchmark_results.png - основные графики")
    print("- detailed_benchmark_analysis.png - детальный анализ")
    print("- benchmark_report.txt - текстовый отчет")
    print("\nДля просмотра результатов откройте файлы выше.")

if __name__ == "__main__":
    main() 