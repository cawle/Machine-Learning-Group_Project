# Machine-Learning-Group_Project

# Digital Marketing Campaign Optimization  
### Data Mining & Analytics Project – Group 15

## 🚀 Objectives
- Clean, encode, transform, and normalize a real-world digital marketing dataset.
- Build a complete analytical pipeline including Market Basket Analysis, Clustering, Recommendation, and Anomaly Detection.
- Evaluate performance using support, confidence, lift, inertia, Precision@5, Recall@5, and anomaly scores.

---

## 🔧 Project Scope

### **1. Data Preparation**
- Handling missing values  
- Removing duplicates  
- Encoding categorical variables  
- Normalizing numerical features  
- Feature engineering (total purchases, spending, campaign responses, family size)

---

### **2. Market Basket Analysis (Apriori)**
- Identification of frequent itemsets  
- Generation of association rules  
- Strong links found among meats, wines, and premium items  
- Supports product bundling, cross-selling, and store layout strategies  

---

### **3. Customer Segmentation (K-Means)**
Four customer clusters identified:
1. High-Value Customers  
2. Active Shoppers  
3. At-Risk Customers  
4. Budget Families  

Each segment enables targeted and personalized marketing campaigns.

---

### **4. Recommender System (Hybrid)**
- Hybrid = content-based + collaborative filtering  
- Balanced precision and recall  
- Outperforms individual recommendation methods  

---

### **5. Anomaly Detection (Isolation Forest)**
- Detects unusual customer spending and behavior  
- Identifies fraud-like and VIP customer patterns  
- PCA used for visual separation  

---

## 📊 Key Findings

### **Market Basket Analysis**
- Support up to 0.64, confidence 0.80–0.89  
- Strong associations among meats, wines, and premium products  

### **Clustering**
- Even distribution across four clusters  
- Clear behavioral differences between clusters  

### **Recommender System**
- Hybrid method achieved best overall performance  

### **Anomaly Detection**
- Small cluster of high-value or suspicious customers  
- Useful for fraud detection and engagement strategies  

---

## 🧠 Methodology (KDD)
1. Data Selection  
2. Data Preprocessing  
3. Data Transformation  
4. Data Mining  
5. Knowledge Presentation  

---

## 🛠 Technologies Used
- Python  
- Scikit-learn  
- Pandas & NumPy  
- MLxtend  
- Matplotlib & Seaborn  
- Jupyter Notebook  

---


## 📌 Future Work
- Real-time campaign monitoring  
- Deep learning models (NCF, Transformers, LSTMs)  
- Explainable AI for transparent recommendations  
- Expansion to healthcare, finance, education  
