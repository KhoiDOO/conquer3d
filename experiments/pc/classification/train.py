import torch
import torch.nn as nn
from tqdm.auto import tqdm
import os
import json

from conquer3d.data.dataset.digit3d import PointDigit3D
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from experiments.pc.classification.models import PointTransformerCls

def main():
    print("Initializing Datasets...")
    train_dataset = PointDigit3D(root="~/.conquer3d/", train=True, download=False, cached=True, num_points=512)
    test_dataset = PointDigit3D(root="~/.conquer3d/", train=False, download=False, cached=True, num_points=512)

    # Standard default collate_fn since size is fixed [512, 6]
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=8,
        persistent_workers=True,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=8,
        persistent_workers=True,
        pin_memory=True
    )

    print("Initializing Model...")
    model = PointTransformerCls(
        depth=4, 
        in_channels=6, 
        num_classes=10, 
        dim=128, 
        share_planes=8, 
        patch_size=32
    ).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    num_epochs = 20
    warmup_epochs = 5

    def lr_lambda(epoch):
        import math
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    print("Starting Training Loop...")
    history = []
    best_test_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for points, features, labels in progress_bar:
            # features contains [points, normals]
            features = features.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(features)
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}", 
                'Acc': f"{100.*correct/total:.2f}%"
            })

        # Evaluate on Test Set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for points, features, labels in test_loader:
                features = features.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(features)
                    
                _, predicted = logits.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        test_acc = 100. * test_correct / test_total
        
        print(f"==> Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "point_classification.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[*] Saved new best model to {ckpt_path} with test acc: {best_test_acc:.2f}%")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc
        })
        
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "point_classification.json")
        with open(json_path, "w") as f:
            json.dump(history, f, indent=4)
            
        scheduler.step()

    print("Training finished.")

if __name__ == "__main__":
    main()
