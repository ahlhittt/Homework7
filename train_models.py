import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import numpy as np

class ResNet18Trainer:
    def __init__(self, image_size=224, batch_size=32, num_epochs=10):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Инициализация обучения для {image_size}x{image_size}")
        print(f"   Устройство: {self.device}")
        print(f"   Размер батча: {batch_size}")
        print(f"   Количество эпох: {num_epochs}")
        
        # Создаем директорию для весов
        os.makedirs('./weights', exist_ok=True)
        
        # Определяем трансформации
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Загружаем модель
        self.model = torchvision.models.resnet18(pretrained=True)
        num_classes = 10  # CIFAR-10
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)
        
        # Загружаем данные
        self.train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform
        )
        self.test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.transform
        )
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        # Критерий и оптимизатор
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar для батчей
        batch_pbar = tqdm(self.train_loader, desc=f'Batches', unit='batch', 
                         position=1, leave=False)
        
        for batch_idx, (data, target) in enumerate(batch_pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Обновляем progress bar с текущими метриками
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            batch_pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        batch_pbar.close()
            
        return running_loss / len(self.train_loader), 100. * correct / total
    
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        # Progress bar для тестирования
        test_pbar = tqdm(self.test_loader, desc='Testing', unit='batch', 
                        position=1, leave=False)
        
        with torch.no_grad():
            for data, target in test_pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Обновляем progress bar
                current_loss = test_loss / (test_pbar.n + 1)
                current_acc = 100. * correct / total
                test_pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        test_pbar.close()
        return test_loss / len(self.test_loader), 100. * correct / total
    
    def train(self):
        best_acc = 0
        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        
        # Progress bar для эпох
        epoch_pbar = tqdm(range(self.num_epochs), desc=f'Training {self.image_size}x{self.image_size}', 
                         unit='epoch', position=0, leave=True)
        
        for epoch in epoch_pbar:
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.test()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            # Обновляем progress bar с информацией
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Train Acc': f'{train_acc:.2f}%',
                'Test Loss': f'{test_loss:.4f}',
                'Test Acc': f'{test_acc:.2f}%',
                'Best': f'{best_acc:.2f}%'
            })
            
            # Сохраняем лучшую модель
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), 
                          f'./weights/best_resnet18_{self.image_size}.pth')
        
        epoch_pbar.close()
        
        # Сохраняем историю обучения
        with open(f'./weights/training_history_{self.image_size}.json', 'w') as f:
            json.dump(history, f)
        
        return best_acc

def main():
    image_sizes = [224, 256, 384, 512]
    
    # Общий progress bar для всех моделей
    model_pbar = tqdm(image_sizes, desc='Training Models', unit='model')
    
    for size in model_pbar:
        model_pbar.set_description(f'Training {size}x{size}')
        
        trainer = ResNet18Trainer(image_size=size, batch_size=32, num_epochs=5)
        best_acc = trainer.train()
        
        model_pbar.set_postfix({'Best Acc': f'{best_acc:.2f}%'})
        
        print(f"\nМодель {size}x{size} обучена!")
        print(f"   Лучшая точность: {best_acc:.2f}%")
        print(f"   Сохранена в: ./weights/best_resnet18_{size}.pth")
    
    model_pbar.close()
    print(f"\nВсе модели обучены успешно!")

if __name__ == "__main__":
    main() 