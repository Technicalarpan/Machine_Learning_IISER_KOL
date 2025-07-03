
<h1 align="center">🧠 Machine Learning Fine-Tuning Project</h1>

<div align="center">
  <strong>Compare Base, Fully Fine-Tuned, and K-Shot Models</strong><br/>
  📁 Based on a custom dataset | 🧪 Built with Scikit-learn | 👨‍💻 Author: <b>Arpan Mukherjee</b>
</div>

---

## 📌 Overview

This project demonstrates the **impact of fine-tuning** on machine learning models trained on a dataset. It includes:

- ✅ **Baseline Training**
- 🔄 **Full Fine-Tuning**
- 🎯 **Few-Shot (K-Shot) Fine-Tuning**

Each approach is implemented using separate Python scripts and evaluated on the same dataset.

---

## 📂 Project Structure

```bash
iiser/
├── data.csv                    # Custom dataset for training/evaluation
├── ml_beforefinetunned.py     # Model training without fine-tuning
├── ml_finetunned.py           # Fully fine-tuned model
├── ml_kshotfinetunned.py      # K-shot fine-tuning approach
└── README.md                  # This file
```

---

## 💡 Features

- 🔍 Analyze model performance before & after tuning  
- 🧪 Compare different fine-tuning methods  
- 📈 Uses **scikit-learn** for ML implementation  
- 💾 Easy-to-modify dataset (`data.csv`)  

---

## 🛠️ Tech Stack

| Technology       | Description                   |
|------------------|-------------------------------|
| **Python**       | Programming Language           |
| **Pandas**       | Data Manipulation              |
| **NumPy**        | Numerical Computing            |
| **Scikit-learn** | ML Algorithms & Evaluation     |

---

## 🚀 How to Run

1. **Install dependencies**  
```bash
pip install pandas numpy scikit-learn
```

2. **Run any script**  
Make sure you are inside the `iiser/` folder.

```bash
# Baseline
python ml_beforefinetunned.py

# Full Fine-Tuning
python ml_finetunned.py

# K-Shot Fine-Tuning
python ml_kshotfinetunned.py
```

---

## 📊 Dataset Info

- File: `data.csv`  
- Format: Standard CSV with features + target labels  
- You can replace it with your own dataset for experimentation.

---

## 🧑‍💻 Author

**Made by: Arpan Mukherjee**  
🌐 [GitHub Profile](https://github.com/Technicalarpan)

---

## 📜 License

This project is intended for **educational and research purposes** only.

---
