import sys
import io
import os

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import time

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

FASHION_CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

class FashionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.block3(self.block2(self.block1(x))))

DIGIT_CLASS_NAMES = [str(d) for d in range(10)]

class DigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

SENTIMENT_CLASS_NAMES = ["negative", "positive"]

_POSITIVE_TEXTS = [
    "this product is amazing and wonderful",
    "absolutely love it fantastic quality",
    "great service highly recommend",
    "excellent work very satisfied",
    "best purchase i have ever made",
    "wonderful experience will buy again",
    "outstanding quality and fast delivery",
    "perfect fit and great material",
    "love the design it is beautiful",
    "superb craftsmanship really impressed",
    "this is incredible and high quality",
    "very happy with this purchase",
    "the product exceeded my expectations",
    "brilliant quality and fast shipping",
    "delightful and well-made product",
    "amazing value for the price",
    "top notch service and quality",
    "really satisfied with this item",
    "this is exactly what i needed",
    "highly recommend to everyone",
    "the best product in its category",
    "the quality is exceptional and premium",
    "such a great deal i love it",
    "awesome features and easy to use",
    "works perfectly and looks amazing",
]

_NEGATIVE_TEXTS = [
    "terrible quality would not recommend",
    "very disappointed with this product",
    "worst purchase ever waste of money",
    "broke after one day horrible",
    "do not buy this awful product",
    "very poor quality and bad design",
    "completely useless returned immediately",
    "the product is defective and cheap",
    "horrible experience never again",
    "waste of time and money",
    "extremely bad quality avoid this",
    "not as described very misleading",
    "product stopped working after a week",
    "terrible customer service and slow shipping",
    "the worst product i have ever used",
    "falls apart easily very fragile",
    "does not work as advertised",
    "poor build quality and ugly design",
    "i regret buying this product",
    "absolute garbage stay away",
    "poorly made and overpriced",
    "this product is a total scam",
    "arrived broken and customer support ignored me",
    "flimsy material and bad stitching",
    "never buying from this brand again",
]

VOCAB_SIZE = 512

def _build_vocab(texts):
    from collections import Counter
    import re
    words = []
    for t in texts:
        words.extend(re.findall(r'\b\w+\b', t.lower()))
    most_common = [w for w, _ in Counter(words).most_common(VOCAB_SIZE)]
    vocab = {w: i for i, w in enumerate(most_common)}
    return vocab

def _text_to_bow(text: str, vocab: dict) -> np.ndarray:
    import re
    vec = np.zeros(VOCAB_SIZE, dtype=np.float32)
    for w in re.findall(r'\b\w+\b', text.lower()):
        idx = vocab.get(w)
        if idx is not None:
            vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

class SentimentMLP(nn.Module):
    def __init__(self, input_size=VOCAB_SIZE, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def _save_onnx(model, dummy, path, input_names, output_names, dynamic_axes):
    model.eval()
    torch.onnx.export(
        model, dummy, path,
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"  Saved ONNX  : {path}")

def _save_baseline(model, loader, class_names, num_classes, path, device, model_name, version):
    model.eval()
    all_preds, all_confs = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            mc, pr = probs.max(1)
            all_preds.extend(pr.cpu().numpy().tolist())
            all_confs.extend(mc.cpu().numpy().tolist())

    dist = (np.bincount(all_preds, minlength=num_classes) / len(all_preds)).tolist()
    conf_hist, _ = np.histogram(all_confs, bins=20, range=(0, 1))
    baseline = {
        "model_name": model_name,
        "class_distribution": dist,
        "confidence_distribution": (conf_hist / conf_hist.sum()).tolist(),
        "mean_confidence": float(np.mean(all_confs)),
        "std_confidence": float(np.std(all_confs)),
        "class_names": class_names,
        "num_classes": num_classes,
        "num_samples": len(all_preds),
        "model_version": version,
    }
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"  Baseline    : {path}")

def _train_vision_model(
    model, train_loader, test_loader, device,
    epochs, tag, save_prefix, class_names, num_classes, version,
):
    print(f"\n{'=' * 60}")
    print(f" Training {tag}")
    print(f"{'=' * 60}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        correct = total = 0
        loss_sum = 0.0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        scheduler.step()

        model.eval()
        tc = tt = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                _, pred = model(imgs).max(1)
                tt += labels.size(0)
                tc += pred.eq(labels).sum().item()
        test_acc = 100.0 * tc / tt
        print(
            f"  Epoch [{epoch+1:2d}/{epochs}] "
            f"Loss={loss_sum/len(train_loader):.4f}  "
            f"TrainAcc={100.0*correct/total:.2f}%  "
            f"TestAcc={test_acc:.2f}%  "
            f"({time.time()-t0:.1f}s)"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{save_prefix}_best.pth"))

    model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, f"{save_prefix}_best.pth"), weights_only=True)
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "num_classes": num_classes,
        "accuracy": best_acc,
        "model_version": version,
    }, os.path.join(MODELS_DIR, f"{save_prefix}.pth"))
    print(f"  Saved PyTorch: {os.path.join(MODELS_DIR, f'{save_prefix}.pth')}  (best acc {best_acc:.2f}%)")

    dummy = torch.randn(1, 1, 28, 28).to(device)
    _save_onnx(
        model, dummy, os.path.join(MODELS_DIR, f"{save_prefix}.onnx"),
        ["input"], ["output"],
        {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    _save_baseline(
        model, test_loader, class_names, num_classes,
        os.path.join(MODELS_DIR, f"{save_prefix}_baseline.json"), device, tag, version,
    )
    return model

def _download_and_load_sentiment_data():
    import urllib.request
    import zipfile
    import io
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip"
    texts = []
    labels = []
    
    print("  Downloading UCI Sentiment Labelled Sentences dataset...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            zip_data = response.read()
        
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            files_to_read = [
                "sentiment labelled sentences/amazon_cells_labelled.txt",
                "sentiment labelled sentences/imdb_labelled.txt",
                "sentiment labelled sentences/yelp_labelled.txt"
            ]
            for file_path in files_to_read:
                if file_path in z.namelist():
                    with z.open(file_path) as f:
                        content = f.read().decode('utf-8')
                        for line in content.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split('\t')
                            if len(parts) == 2:
                                text, label_str = parts[0], parts[1]
                                try:
                                    label = int(label_str.strip())
                                    if label in (0, 1) and len(text.strip()) > 0:
                                        texts.append(text.strip())
                                        labels.append(label)
                                except ValueError:
                                    pass
        print(f"  Loaded {len(texts)} samples from UCI Sentiment Dataset.")
        return texts, labels
    except Exception as e:
        print(f"  Failed to load online dataset: {e}")
        print("  Falling back to local hardcoded sample sentences.")
        fallback_texts = _POSITIVE_TEXTS + _NEGATIVE_TEXTS
        fallback_labels = [1] * len(_POSITIVE_TEXTS) + [0] * len(_NEGATIVE_TEXTS)
        return fallback_texts, fallback_labels

def train_all_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    fashion_train = datasets.FashionMNIST("./data", train=True, download=True, transform=tf_train)
    fashion_test  = datasets.FashionMNIST("./data", train=False, download=True, transform=tf_test)
    _train_vision_model(
        FashionCNN(), DataLoader(fashion_train, 128, shuffle=True, num_workers=0),
        DataLoader(fashion_test, 256, num_workers=0), device,
        epochs=15, tag="FashionMNIST", save_prefix="fashion_mnist",
        class_names=FASHION_CLASS_NAMES, num_classes=10, version="v2.0.0",
    )

    tf_digit = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    digit_train = datasets.MNIST("./data", train=True, download=True, transform=tf_digit)
    digit_test  = datasets.MNIST("./data", train=False, download=True, transform=tf_digit)
    _train_vision_model(
        DigitCNN(), DataLoader(digit_train, 128, shuffle=True, num_workers=0),
        DataLoader(digit_test, 256, num_workers=0), device,
        epochs=10, tag="MNIST Digits", save_prefix="mnist_digits",
        class_names=DIGIT_CLASS_NAMES, num_classes=10, version="v1.0.0",
    )

    print(f"\n{'=' * 60}")
    print(" Training Sentiment Classifier")
    print(f"{'=' * 60}")

    texts, labels = _download_and_load_sentiment_data()
    vocab = _build_vocab(texts)

    with open(os.path.join(MODELS_DIR, "sentiment_vocab.json"), "w") as f:
        json.dump(vocab, f)
    print(f"  Vocab size={len(vocab)}, saved to {os.path.join(MODELS_DIR, 'sentiment_vocab.json')}")

    X = np.array([_text_to_bow(t, vocab) for t in texts], dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    
    # Adjust batch size and epochs based on dataset size for performance
    batch_size = 64 if len(texts) > 100 else 8
    epochs = 20 if len(texts) > 100 else 80
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sent_model = SentimentMLP().to(device)
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(sent_model.parameters(), lr=0.002)

    for epoch in range(epochs):
        sent_model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(sent_model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % max(1, epochs // 4) == 0:
            sent_model.eval()
            with torch.no_grad():
                all_out = sent_model(torch.from_numpy(X).to(device))
                acc = (all_out.argmax(1).cpu() == torch.from_numpy(y)).float().mean() * 100
            print(f"  Epoch [{epoch+1:3d}/{epochs}] Loss={total_loss/len(loader):.4f}  Acc={acc:.1f}%")

    torch.save({
        "model_state_dict": sent_model.state_dict(),
        "class_names": SENTIMENT_CLASS_NAMES,
        "num_classes": 2,
        "model_version": "v1.0.0",
    }, os.path.join(MODELS_DIR, "sentiment.pth"))
    print(f"  Saved PyTorch: {os.path.join(MODELS_DIR, 'sentiment.pth')}")

    dummy_text = torch.randn(1, VOCAB_SIZE).to(device)
    _save_onnx(
        sent_model, dummy_text, os.path.join(MODELS_DIR, "sentiment.onnx"),
        ["input"], ["output"],
        {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    sent_model.eval()
    with torch.no_grad():
        probs = torch.softmax(sent_model(torch.from_numpy(X).to(device)), dim=1)
        mc, pr = probs.max(1)
    sent_baseline = {
        "model_name": "Sentiment",
        "class_distribution": (np.bincount(pr.cpu().numpy(), minlength=2) / len(y)).tolist(),
        "confidence_distribution": np.histogram(mc.cpu().numpy(), bins=20, range=(0, 1))[0].tolist(),
        "mean_confidence": float(mc.mean()),
        "std_confidence": float(mc.std()),
        "class_names": SENTIMENT_CLASS_NAMES,
        "num_classes": 2,
        "num_samples": len(y),
        "model_version": "v1.0.0",
    }
    s = sum(sent_baseline["confidence_distribution"])
    if s > 0:
        sent_baseline["confidence_distribution"] = [c / s for c in sent_baseline["confidence_distribution"]]
    else:
        sent_baseline["confidence_distribution"] = [0.5] * 20
    with open(os.path.join(MODELS_DIR, "sentiment_baseline.json"), "w") as f:
        json.dump(sent_baseline, f, indent=2)
    print(f"  Baseline    : {os.path.join(MODELS_DIR, 'sentiment_baseline.json')}")

    print(f"\n{'=' * 60}")
    print(" Training completed successfully!")
    print(f"{'=' * 60}")
    print(" Models ready:")
    for name in ["fashion_mnist", "mnist_digits", "sentiment"]:
        onnx_exists = "✅" if os.path.exists(os.path.join(MODELS_DIR, f"{name}.onnx")) else "❌"
        base_exists = "✅" if os.path.exists(os.path.join(MODELS_DIR, f"{name}_baseline.json")) else "❌"
        print(f"   {name:20s}  ONNX {onnx_exists}  Baseline {base_exists}")
    print("\n You can now start the application.\n")

if __name__ == "__main__":
    train_all_models()
