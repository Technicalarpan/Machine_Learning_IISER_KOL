<h1 align="center">🧠 Machine Learning Fine-Tuning Project</h1>

<div align="center">
  <strong>Blood Sugar & Sentence Classification | Zero-Shot, Few-Shot, K-Shot, and Fine-Tuning Comparison</strong><br/>
  📁 Real-World Datasets | 🧪 Built with Scikit-learn | 👨‍💻 Author: <b>Arpan Mukherjee</b>
</div>

---

## 📌 Overview

This project is part of a research internship at IISER Kolkata under the Department of Computational and Data Sciences. It explores the **impact of different fine-tuning strategies** on machine learning models, evaluated through **F1 Score optimization**. The project simulates real-world, low-label learning scenarios using two datasets:

- 🩺 **Binary Classification (Medical)** — Blood sugar prediction
- 💬 **Sentiment Classification (Text)** — Sentence type prediction

Implemented learning setups include:

- ✅ **Baseline (Zero-Shot)**  
- 🔄 **Few-Shot and K-Shot Learning**
- 🎯 **Full Fine-Tuning**

---

## 📂 Project Structure

```bash
iiser/
├── Blood Sugar Model/
│   ├── data.csv
│   ├── ml_beforefinetunned.py
│   ├── ml_finetunned.py
│   └── ml_kshotfinetunned.py
│
├── Sentence Classification Model/
│   ├── data.csv
│   ├── nlp_finetunned.py
│   └── nlp_zeroandfewshot.py
│
└── README.md
```

---

## 💡 Features

- 🔬 Evaluate ML models across 4 training setups: Zero, Few, K-shot, and Full Fine-Tuning
- 📊 Track and compare F1 score improvements for each dataset
- ⚙️ Uses **Random Forest** classifier for robustness and interpretability
- 🧪 Built using **scikit-learn**, **pandas**, and **numpy**

---

## 📈 Results Summary

**Blood Sugar Classification (Binary Dataset)**  
- Zero/Few-Shot: F1 = 0.6034  
- K-Shot (20% labeled): F1 = 0.6667  
- Fine-Tuned: F1 = 0.6405

**Sentiment Classification (Text Dataset)**  
- Zero/Few-Shot: F1 = 0.5241  
- Fine-Tuned: F1 = 0.7120  

Fine-tuning significantly improves model performance, especially with increased labeled data.

---

## 🔍 Why Random Forest?

- 🧠 Handles tabular & text-based features well
- ✅ Reduces overfitting with ensemble learning
- 🔍 Offers feature importance insight
- 💡 Performs well with imbalanced data

---

## 🛠️ Tech Stack

| Technology       | Description                        |
|------------------|------------------------------------|
| **Python**       | Programming Language               |
| **Pandas**       | Data manipulation and analysis     |
| **NumPy**        | Numerical computations             |
| **Scikit-learn** | ML modeling and evaluation tools   |

---

## 🚀 How to Run

1. **Install dependencies**

```bash
pip install pandas numpy scikit-learn
```

2. **Run any script**

```bash
# From Blood Sugar Model folder
python ml_beforefinetunned.py
python ml_kshotfinetunned.py
python ml_finetunned.py

# From Sentence Classification Model folder
python nlp_zeroandfewshot.py
python nlp_finetunned.py
```

---

## 📊 Dataset Info

- `data.csv` files are included in each model folder
- Binary classification data: physiological records (age, BMI, glucose, etc.)
- Sentiment classification data: sentence and its polarity label

---

## 📜 Report & Certificate

- 📄 [Internship Report (PDF)](https://drive.google.com/file/d/1pYffOltpLt8tK3omIJmKhUh6bSXg_Sur/view?usp=sharing)  
- 📑 [Certificate Issued by Professor](https://drive.google.com/file/d/1bkQbJ2FrMVn0Tja2iE8yp8_k0WXlbaSe/view?usp=sharing)

---

## 🧑‍💻 Author

**Arpan Mukherjee**  
🎓 3rd Year Undergraduate, NIT Durgapur  
🌐 [GitHub Profile](https://github.com/Technicalarpan)

---

## 📜 License

This project is intended for **educational and research purposes** only.

---