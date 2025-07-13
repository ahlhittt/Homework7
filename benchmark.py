import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import onnx
import onnxruntime as ort
import numpy as np
import time
import psutil
import GPUtil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BenchmarkRunner:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # Получаем информацию о GPU
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_name = gpus[0].name
                self.gpu_memory = gpus[0].memoryTotal
            else:
                self.gpu_name = "CPU Only"
                self.gpu_memory = 0
        except:
            self.gpu_name = "Unknown GPU"
            self.gpu_memory = 0
    
    def load_model(self, image_size, model_path):
        """Загружает PyTorch модель"""
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def create_dummy_input(self, batch_size, image_size):
        """Создает тестовые данные"""
        return torch.randn(batch_size, 3, image_size, image_size).to(self.device)
    
    def measure_pytorch_performance(self, model, dummy_input, num_runs=100):
        """Измеряет производительность PyTorch модели"""
        times = []
        
        # Прогрев
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Измерения
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
        
        return {
            'mean_time': np.mean(filtered_times),
            'std_time': np.std(filtered_times),
            'fps': 1000.0 / np.mean(filtered_times),
            'time_per_image': np.mean(filtered_times) / dummy_input.shape[0]
        }
    
    def convert_to_onnx(self, model, dummy_input, onnx_path):
        """Конвертирует модель в ONNX формат"""
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=11,
            do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}}
        )
    
    def measure_onnx_performance(self, onnx_path, dummy_input, num_runs=100):
        """Измеряет производительность ONNX модели"""
        if torch.cuda.is_available():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Подготавливаем входные данные
        input_name = session.get_inputs()[0].name
        input_data = dummy_input.cpu().numpy()
        
        times = []
        
        # Прогрев
        for _ in range(10):
            _ = session.run(None, {input_name: input_data})
        
        # Измерения
        for _ in range(num_runs):
            start_time = time.time()
            _ = session.run(None, {input_name: input_data})
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        # Убираем 10% лучших и худших результатов
        times = np.array(times)
        sorted_times = np.sort(times)
        n_remove = int(0.1 * len(times))
        filtered_times = sorted_times[n_remove:-n_remove]
        
        return {
            'mean_time': np.mean(filtered_times),
            'std_time': np.std(filtered_times),
            'fps': 1000.0 / np.mean(filtered_times),
            'time_per_image': np.mean(filtered_times) / dummy_input.shape[0]
        }
    
    def measure_tensorrt_performance(self, model, dummy_input, num_runs=100):
        """Измеряет производительность TensorRT модели (для AMD GPU используем DirectML)"""
        try:
            # Для AMD GPU используем torch.compile с режимом max-autotune
            # Также пробуем DirectML через ONNX Runtime
            compiled_model = torch.compile(model, mode="max-autotune")
            
            times = []
            
            # Прогрев
            for _ in range(10):
                with torch.no_grad():
                    _ = compiled_model(dummy_input)
            
            # Измерения
            for _ in range(num_runs):
                start_time = time.time()
                with torch.no_grad():
                    _ = compiled_model(dummy_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            # Убираем 10% лучших и худших результатов
            times = np.array(times)
            sorted_times = np.sort(times)
            n_remove = int(0.1 * len(times))
            filtered_times = sorted_times[n_remove:-n_remove]
            
            return {
                'mean_time': np.mean(filtered_times),
                'std_time': np.std(filtered_times),
                'fps': 1000.0 / np.mean(filtered_times),
                'time_per_image': np.mean(filtered_times) / dummy_input.shape[0]
            }
        except Exception as e:
            print(f"TensorRT/ROCm оптимизация недоступна: {e}")
            return None
    
    def get_system_info(self):
        """Получает информацию о системе"""
        cpu_info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_total': psutil.virtual_memory().total / (1024**3)  # GB
        }
        
        gpu_info = {
            'name': self.gpu_name,
            'memory_total': self.gpu_memory
        }
        
        return {'cpu': cpu_info, 'gpu': gpu_info}
    
    def run_benchmark(self, image_sizes=[224, 256, 384, 512], batch_sizes=[1, 4, 8, 16, 32]):
        """Запускает полный бенчмарк"""
        print(f"Запуск бенчмарка на {self.gpu_name}")
        print("=" * 60)
        
        system_info = self.get_system_info()
        print(f"CPU: {system_info['cpu']['cpu_count']} ядер, {system_info['cpu']['cpu_freq']:.1f} MHz")
        print(f"GPU: {system_info['gpu']['name']}, {system_info['gpu']['memory_total']} MB")
        print(f"RAM: {system_info['cpu']['memory_total']:.1f} GB")
        print("=" * 60)
        
        results = {}
        
        for image_size in image_sizes:
            print(f"\nТестирование для размера изображения: {image_size}x{image_size}")
            print("-" * 50)
            
            model_path = f'./weights/best_resnet18_{image_size}.pth'
            if not os.path.exists(model_path):
                print(f"Модель {model_path} не найдена. Пропускаем...")
                continue
            
            # Загружаем модель
            model = self.load_model(image_size, model_path)
            
            results[image_size] = {}
            
            for batch_size in batch_sizes:
                print(f"  Батч: {batch_size}")
                
                dummy_input = self.create_dummy_input(batch_size, image_size)
                
                # PyTorch
                pytorch_results = self.measure_pytorch_performance(model, dummy_input)
                results[image_size][batch_size] = {'pytorch': pytorch_results}
                
                # ONNX
                onnx_path = f'./weights/model_{image_size}.onnx'
                self.convert_to_onnx(model, dummy_input, onnx_path)
                onnx_results = self.measure_onnx_performance(onnx_path, dummy_input)
                results[image_size][batch_size]['onnx'] = onnx_results
                
                # TensorRT/ROCm
                tensorrt_results = self.measure_tensorrt_performance(model, dummy_input)
                if tensorrt_results:
                    results[image_size][batch_size]['tensorrt'] = tensorrt_results
                
                print(f"    PyTorch: {pytorch_results['fps']:.1f} FPS")
                print(f"    ONNX: {onnx_results['fps']:.1f} FPS")
                if tensorrt_results:
                    print(f"    TensorRT: {tensorrt_results['fps']:.1f} FPS")
        
        self.results = results
        return results
    
    def save_results(self, filename='benchmark_results.json'):
        """Сохраняет результаты в JSON файл"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Результаты сохранены в {filename}")
    
    def create_plots(self):
        """Создает графики производительности"""
        if not self.results:
            print("Нет результатов для построения графиков")
            return
        
        # Подготавливаем данные для графиков
        plot_data = []
        
        for image_size in self.results:
            for batch_size in self.results[image_size]:
                for method, results in self.results[image_size][batch_size].items():
                    plot_data.append({
                        'image_size': image_size,
                        'batch_size': batch_size,
                        'method': method,
                        'fps': results['fps'],
                        'mean_time': results['mean_time'],
                        'time_per_image': results['time_per_image']
                    })
        
        df = pd.DataFrame(plot_data)
        
        # График 1: FPS vs Размер изображения
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            plt.plot(method_data['image_size'], method_data['fps'], 
                    marker='o', label=method.upper(), linewidth=2)
        plt.xlabel('Размер изображения')
        plt.ylabel('FPS')
        plt.title('FPS vs Размер изображения')
        plt.legend()
        plt.grid(True)
        
        # График 2: FPS vs Размер батча
        plt.subplot(2, 2, 2)
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            plt.plot(method_data['batch_size'], method_data['fps'], 
                    marker='s', label=method.upper(), linewidth=2)
        plt.xlabel('Размер батча')
        plt.ylabel('FPS')
        plt.title('FPS vs Размер батча')
        plt.legend()
        plt.grid(True)
        
        # График 3: Ускорение относительно PyTorch
        plt.subplot(2, 2, 3)
        pytorch_fps = df[df['method'] == 'pytorch'].set_index(['image_size', 'batch_size'])['fps']
        
        for method in ['onnx', 'tensorrt']:
            if method in df['method'].unique():
                method_data = df[df['method'] == method]
                method_data = method_data.set_index(['image_size', 'batch_size'])
                speedup = method_data['fps'] / pytorch_fps
                plt.plot(speedup.index.get_level_values(0), speedup.values, 
                        marker='^', label=f'{method.upper()} Speedup', linewidth=2)
        
        plt.xlabel('Размер изображения')
        plt.ylabel('Ускорение')
        plt.title('Ускорение относительно PyTorch')
        plt.legend()
        plt.grid(True)
        
        # График 4: Время на изображение
        plt.subplot(2, 2, 4)
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            plt.plot(method_data['image_size'], method_data['time_per_image'], 
                    marker='d', label=method.upper(), linewidth=2)
        plt.xlabel('Размер изображения')
        plt.ylabel('Время на изображение (мс)')
        plt.title('Время обработки одного изображения')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Графики сохранены в benchmark_results.png")

def main():
    benchmark = BenchmarkRunner()
    results = benchmark.run_benchmark()
    benchmark.save_results()
    benchmark.create_plots()
    
    # Выводим сводку результатов
    print("\n" + "="*60)
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print("="*60)
    
    for image_size in results:
        print(f"\nРазмер изображения: {image_size}x{image_size}")
        print("-" * 40)
        
        for batch_size in results[image_size]:
            print(f"  Батч: {batch_size}")
            pytorch_fps = results[image_size][batch_size]['pytorch']['fps']
            
            for method, data in results[image_size][batch_size].items():
                speedup = data['fps'] / pytorch_fps if method != 'pytorch' else 1.0
                print(f"    {method.upper()}: {data['fps']:.1f} FPS (ускорение: {speedup:.2f}x)")

if __name__ == "__main__":
    main() 