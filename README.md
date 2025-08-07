# 💧 Water Quality Prediction

A machine learning-based web application that classifies the quality of water using real-world data into categories such as **Safe for Immediate Consumption**, **Good**, and **Poor**, based on physicochemical and biological parameters.

> 🔬 Built during a 45-day internship at **C-DAC Mohali**

---

<img width="1909" height="854" alt="image" src="https://github.com/user-attachments/assets/03117099-adf0-485d-91a5-894062fd86e2" />


<img width="1896" height="867" alt="image" src="https://github.com/user-attachments/assets/53885713-6353-485f-87f7-d33036ed6e85" />

---


The project includes:
- A **real-world water quality dataset** containing physicochemical test values collected from various sources
- A **machine learning pipeline** developed in **Google Colab (`.ipynb`)**
- An **interactive web application** built using **Streamlit (`app.py`)** that allows users to upload water testing data and get predictions about water quality

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `Project_Water_Quality_Analysis.ipynb` | Jupyter Notebook for data cleaning, visualization, model training, and evaluation |
| `app.py` | Streamlit app for interactive water quality classification |
| `Water Quality Analysis Project Dataset.xlsx` | Water quality dataset used for analysis and model training |
| `requirements.txt` | List of Python libraries required to run the project |
| `README.md` | Project overview and usage instructions |

---

## 🔍 Features

- Upload Excel files with water sample data
- Automatically clean, preprocess, and extract key features
- Classify water quality into:
  - ✅ Safe for Immediate Consumption
  - 👍 Good
  - ⚠️ Poor
- Visualize the class distribution
- Download predictions as `.csv`
- Manual input mode for quick testing

---

## 🧪 Dataset & Features

- 📍 **Source**: Central Pollution Control Board (CPCB)

The application processes physicochemical parameters of water including:

- **pH**
- **Temperature**
- **Dissolved Oxygen (DO)**
- **Biological Oxygen Demand (BOD)**
- **Nitrate (NO3)**
- **Conductivity (Cond)**
- **Total Coliform (Tc)**
- **Fecal Coliform (Fc)**

- ➕ Log-transformation was used on TC and FC to reduce skewness and improve model performance.

These features are averaged from min/max columns and preprocessed for classification.

---

## 🧰 Tech Stack / Libraries Used

### 🧮 Machine Learning:
- `RandomForestClassifier` (Sklearn)
- `StandardScaler`, `LabelEncoder`, `train_test_split`

### 📊 Data Handling:
- `pandas`, `numpy`

### 📈 Visualization:
- `matplotlib`, `seaborn`

### 🖥️ Web App:
- `Streamlit` for frontend interface

---

📊 Streamlit App Features
Upload Excel files with water test data

Automatic cleaning, normalization, and classification

Visualizations:

Distribution of water quality classes

Basic statistics preview

Manual input interface to predict custom water quality results

CSV download of prediction result

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Sehajkaur24/Water-quality-prediction.git
   cd Water-quality-prediction
   ```
2. Install dependencies:

```bash

pip install -r requirements.txt
```
3. Run the Streamlit app:

```bash

streamlit run app.py
```
