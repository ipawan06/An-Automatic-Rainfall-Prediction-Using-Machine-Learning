# Automatic Rainfall Prediction Using Machine Learning

## 📌 Overview
This project leverages machine learning and deep learning techniques to predict rainfall using historical weather data (1901–2015). Accurate rainfall prediction is vital for agriculture, water resource management, and disaster preparedness. The study compares **ARIMA** and **ANN (Artificial Neural Network)** models and proposes a hybrid approach to achieve high predictive accuracy and efficiency.  

## 🛠 Tools & Technologies
- **Languages**: Python (3.6+)  
- **Frameworks/Libraries**: scikit-learn, Keras, NumPy, pandas, Matplotlib  
- **Models Used**:  
  - ARIMA (AutoRegressive Integrated Moving Average)  
  - ANN (Multi-Layer Perceptron & Radial Basis Function)  
  - Hybrid ARIMA + LSTM (experimental)  
- **Platform**: Anaconda, Tkinter (for desktop app interface)  

## 🔍 Approach
1. **Data Collection & Preprocessing**
   - Daily and annual rainfall data from **33 districts & 400+ rain-gauge stations (1957–2017)**.
   - Cleaned noisy data, removed missing values, normalized features.

2. **Feature Engineering**
   - Lagged rainfall values (t-1, t-2, t-3) as predictors.
   - Geographic features (latitude, longitude) for localization.

3. **Modeling**
   - **ANN Models**:
     - MLP Backpropagation network (4 inputs → hidden layer → output).
     - Radial Basis Function (RBF) model for rainfall forecasting.  
   - **ARIMA Model**:
     - Time series analysis for monthly/annual rainfall forecasting.  
   - **Hybrid Model**:
     - Explored ARIMA + LSTM for improved accuracy.  

4. **System Design**
   - Built modular pipeline: Data preprocessing → Training → Prediction → Visualization.  
   - Developed Tkinter-based desktop app for interactive rainfall forecasting.  

5. **Evaluation**
   - Metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R² Score.  
   - Compared ARIMA vs ANN vs Hybrid models.  

## 📊 Key Insights
- **ANN achieved ~99.6% accuracy** for short-term rainfall prediction.  
- **ARIMA excelled at time-series forecasting** for long-term rainfall trends.  
- Combining ML + DL improved reliability and robustness.  
- The system can support applications in **agriculture planning, water resource allocation, and disaster preparedness**.  

## 🚀 Results
- Delivered rainfall forecasts with high accuracy and scalability.  
- Built a **desktop application** for real-time rainfall prediction.  
- Improved **resource efficiency** and **decision-making capabilities** for climate-sensitive sectors.  


---

## 👤 Author
**Pawan Kalyan Ramisetty Narayanaswamy**  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/pawan6/)  
📧 006pawank@gmail.com  
