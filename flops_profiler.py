import torch
import torch.nn as nn
import torchvision
import numpy as np
import time
import psutil
import GPUtil
from thop import profile
import matplotlib.pyplot as plt
import json
import os

class FLOPSProfiler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Получаем информацию о системе
        self.system_info = self.get_system_info()
        
    def get_system_info(self):
        """Получает информацию о системе"""
        cpu_info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_total': psutil.virtual_memory().total / (1024**3)  # GB
        }
        
        gpu_info = {}
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_info = {
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_free': gpu.memoryFree,
                    'temperature': gpu.temperature,
                    'load': gpu.load * 100 if gpu.load else 0
                }
            else:
                gpu_info = {'name': 'CPU Only'}
        except:
            gpu_info = {'name': 'Unknown GPU'}
        
        return {'cpu': cpu_info, 'gpu': gpu_info}
    
    def load_model(self, image_size, model_path):
        """Загружает модель"""
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def calculate_flops(self, model, input_size):
        """Вычисляет количество FLOPS для модели"""
        dummy_input = torch.randn(1, 3, input_size, input_size).to(self.device)
        
        try:
            # Используем thop для подсчета FLOPS
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            return flops, params
        except Exception as e:
            print(f"Ошибка при подсчете FLOPS: {e}")
            return 0, 0
    
    def measure_theoretical_performance(self):
        """Измеряет теоретическую производительность системы"""
        theoretical_performance = {}
        
        # Теоретическая производительность CPU (примерно)
        cpu_freq_ghz = self.system_info['cpu']['cpu_freq'] / 1000
        theoretical_performance['cpu_flops'] = cpu_freq_ghz * 1e9 * self.system_info['cpu']['cpu_count'] * 2  # 2 FLOPS per cycle per core
        
        # Теоретическая производительность GPU (примерно)
        if 'name' in self.system_info['gpu'] and self.system_info['gpu']['name'] != 'CPU Only':
            # Примерные значения для разных GPU
            gpu_name = self.system_info['gpu']['name'].lower()
            
            if 'rx' in gpu_name or 'radeon' in gpu_name:
                # AMD GPU
                if '6900' in gpu_name or '6800' in gpu_name:
                    theoretical_performance['gpu_flops'] = 23.04e12  # RX 6900 XT
                elif '6700' in gpu_name:
                    theoretical_performance['gpu_flops'] = 13.21e12  # RX 6700 XT
                elif '6600' in gpu_name:
                    theoretical_performance['gpu_flops'] = 7.19e12   # RX 6600 XT
                else:
                    theoretical_performance['gpu_flops'] = 5e12  # Примерное значение
            elif 'rtx' in gpu_name or 'gtx' in gpu_name:
                # NVIDIA GPU
                if '4090' in gpu_name:
                    theoretical_performance['gpu_flops'] = 83e12
                elif '4080' in gpu_name:
                    theoretical_performance['gpu_flops'] = 49e12
                elif '4070' in gpu_name:
                    theoretical_performance['gpu_flops'] = 29e12
                else:
                    theoretical_performance['gpu_flops'] = 10e12  # Примерное значение
            else:
                theoretical_performance['gpu_flops'] = 5e12  # Примерное значение
        else:
            theoretical_performance['gpu_flops'] = 0
        
        return theoretical_performance
    
    def measure_actual_performance(self, model, input_size, num_runs=100):
        """Измеряет фактическую производительность"""
        dummy_input = torch.randn(1, 3, input_size, input_size).to(self.device)
        
        # Прогрев
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Измерения
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # в миллисекундах
        
        # Убираем 10% лучших и худших результатов
        times = np.array(times)
        sorted_times = np.sort(times)
        n_remove = int(0.1 * len(times))
        filtered_times = sorted_times[n_remove:-n_remove]
        
        avg_time = np.mean(filtered_times)
        fps = 1000.0 / avg_time
        
        return {
            'avg_time': avg_time,
            'fps': fps,
            'std_time': np.std(filtered_times)
        }
    
    def calculate_utilization(self, flops, time_seconds, theoretical_flops):
        """Вычисляет загруженность железа"""
        if theoretical_flops == 0:
            return 0
        
        actual_flops_per_second = flops / time_seconds
        utilization = (actual_flops_per_second / theoretical_flops) * 100
        return min(utilization, 100)  # Ограничиваем до 100%
    
    def profile_models(self, image_sizes=[224, 256, 384, 512]):
        """Профилирует все модели"""
        print("Профилирование FLOPS и загруженности железа")
        print("=" * 60)
        
        theoretical_performance = self.measure_theoretical_performance()
        
        print(f"Теоретическая производительность:")
        print(f"CPU: {theoretical_performance['cpu_flops']/1e12:.2f} TFLOPS")
        if theoretical_performance['gpu_flops'] > 0:
            print(f"GPU: {theoretical_performance['gpu_flops']/1e12:.2f} TFLOPS")
        print()
        
        results = {}
        
        for image_size in image_sizes:
            print(f"Анализ модели для размера {image_size}x{image_size}")
            print("-" * 40)
            
            model_path = f'./weights/best_resnet18_{image_size}.pth'
            model = self.load_model(image_size, model_path)
            
            # Вычисляем FLOPS
            flops, params = self.calculate_flops(model, image_size)
            print(f"FLOPS: {flops/1e9:.2f} GFLOPS")
            print(f"Параметры: {params/1e6:.2f} M")
            
            # Измеряем производительность
            perf_results = self.measure_actual_performance(model, image_size)
            print(f"FPS: {perf_results['fps']:.1f}")
            print(f"Время: {perf_results['avg_time']:.2f} мс")
            
            # Вычисляем загруженность
            time_seconds = perf_results['avg_time'] / 1000
            cpu_utilization = self.calculate_utilization(flops, time_seconds, theoretical_performance['cpu_flops'])
            gpu_utilization = 0
            
            if theoretical_performance['gpu_flops'] > 0:
                gpu_utilization = self.calculate_utilization(flops, time_seconds, theoretical_performance['gpu_flops'])
                print(f"Загруженность GPU: {gpu_utilization:.1f}%")
            
            print(f"Загруженность CPU: {cpu_utilization:.1f}%")
            print()
            
            results[image_size] = {
                'flops': flops,
                'params': params,
                'fps': perf_results['fps'],
                'time': perf_results['avg_time'],
                'cpu_utilization': cpu_utilization,
                'gpu_utilization': gpu_utilization
            }
        
        return results, theoretical_performance
    
    def create_utilization_plots(self, results, theoretical_performance):
        """Создает графики загруженности"""
        image_sizes = list(results.keys())
        cpu_utils = [results[size]['cpu_utilization'] for size in image_sizes]
        gpu_utils = [results[size]['gpu_utilization'] for size in image_sizes]
        flops_values = [results[size]['flops']/1e9 for size in image_sizes]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Анализ FLOPS и загруженности железа', fontsize=16, fontweight='bold')
        
        # График 1: FLOPS vs Размер изображения
        axes[0, 0].plot(image_sizes, flops_values, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Размер изображения')
        axes[0, 0].set_ylabel('FLOPS (GFLOPS)')
        axes[0, 0].set_title('FLOPS vs Размер изображения')
        axes[0, 0].grid(True, alpha=0.3)
        
        # График 2: Загруженность CPU
        axes[0, 1].plot(image_sizes, cpu_utils, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Размер изображения')
        axes[0, 1].set_ylabel('Загруженность CPU (%)')
        axes[0, 1].set_title('Загруженность CPU')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 100)
        
        # График 3: Загруженность GPU
        if any(gpu_utils):
            axes[1, 0].plot(image_sizes, gpu_utils, 'go-', linewidth=2, markersize=8)
            axes[1, 0].set_xlabel('Размер изображения')
            axes[1, 0].set_ylabel('Загруженность GPU (%)')
            axes[1, 0].set_title('Загруженность GPU')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim(0, 100)
        else:
            axes[1, 0].text(0.5, 0.5, 'GPU недоступен', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Загруженность GPU')
        
        # График 4: Сравнение загруженности
        x = np.arange(len(image_sizes))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, cpu_utils, width, label='CPU', alpha=0.8)
        if any(gpu_utils):
            axes[1, 1].bar(x + width/2, gpu_utils, width, label='GPU', alpha=0.8)
        
        axes[1, 1].set_xlabel('Размер изображения')
        axes[1, 1].set_ylabel('Загруженность (%)')
        axes[1, 1].set_title('Сравнение загруженности CPU vs GPU')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(image_sizes)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig('flops_utilization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Графики FLOPS и загруженности сохранены в flops_utilization_analysis.png")
    
    def save_results(self, results, theoretical_performance, filename='flops_analysis.json'):
        """Сохраняет результаты анализа"""
        output = {
            'system_info': self.system_info,
            'theoretical_performance': theoretical_performance,
            'results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"Результаты анализа FLOPS сохранены в {filename}")
    
    def print_summary(self, results, theoretical_performance):
        """Выводит сводку результатов"""
        print("\n" + "="*60)
        print("СВОДКА АНАЛИЗА FLOPS И ЗАГРУЖЕННОСТИ")
        print("="*60)
        
        print(f"Система: {self.system_info['gpu']['name']}")
        print(f"CPU: {self.system_info['cpu']['cpu_count']} ядер, {self.system_info['cpu']['cpu_freq']:.1f} MHz")
        print(f"RAM: {self.system_info['cpu']['memory_total']:.1f} GB")
        print()
        
        print("Теоретическая производительность:")
        print(f"CPU: {theoretical_performance['cpu_flops']/1e12:.2f} TFLOPS")
        if theoretical_performance['gpu_flops'] > 0:
            print(f"GPU: {theoretical_performance['gpu_flops']/1e12:.2f} TFLOPS")
        print()
        
        print("Результаты по размерам изображений:")
        for size in results:
            result = results[size]
            print(f"{size}x{size}:")
            print(f"  FLOPS: {result['flops']/1e9:.2f} GFLOPS")
            print(f"  FPS: {result['fps']:.1f}")
            print(f"  Загруженность CPU: {result['cpu_utilization']:.1f}%")
            if result['gpu_utilization'] > 0:
                print(f"  Загруженность GPU: {result['gpu_utilization']:.1f}%")
            print()

def main():
    profiler = FLOPSProfiler()
    results, theoretical_performance = profiler.profile_models()
    profiler.create_utilization_plots(results, theoretical_performance)
    profiler.save_results(results, theoretical_performance)
    profiler.print_summary(results, theoretical_performance)

if __name__ == "__main__":
    main() 