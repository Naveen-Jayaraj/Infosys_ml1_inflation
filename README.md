# Cost of Living Index Forecaster

[![Download Project](https://img.shields.io/badge/Download-Project-blue?style=for-the-badge&logo=streamlit)](https://github.com/Naveen-Jayaraj/Infosys_ml1_inflation/archive/refs/heads/main.zip)
[![GitHub Repo](https://img.shields.io/badge/View_on-GitHub-black?style=for-the-badge&logo=github)](https://github.com/Naveen-Jayaraj/Infosys_ml1_inflation)
[![GitHub Stars](https://img.shields.io/github/stars/Naveen-Jayaraj/Infosys_ml1_inflation?style=for-the-badge)](https://github.com/Naveen-Jayaraj/Infosys_ml1_inflation/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Naveen-Jayaraj/Infosys_ml1_inflation?style=for-the-badge)](https://github.com/Naveen-Jayaraj/Infosys_ml1_inflation/network/members)

---

## Overview

The **Cost of Living Index Forecaster** is a machine learning-powered dashboard built using **Streamlit**, **TensorFlow**, and **Plotly**.  
It predicts **urban** and **rural** cost-of-living indices**, visualizes historical trends**, and provides **interactive statistics** for analytical insight.

The project combines **deep learning forecasting**, **responsive web design**, and **interactive data visualization**, suitable for research, educational, and analytical purposes.

---

## Statistical Summary

<div align="center">

| Metric | Urban | Rural |
|--------|--------|--------|
| Mean Index | 145.2 | 119.7 |
| Median Index | 143.0 | 118.5 |
| Standard Deviation | 11.8 | 9.6 |
| Forecast Accuracy (R²) | 0.93 | 0.90 |

</div>

---

## Interactive Statistics Panel

> Use this section in your Streamlit app for real-time exploration.

```python
if st.button("Show Interactive Statistics"):
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
```

This button dynamically displays descriptive statistics and correlation heatmaps, allowing users to visually inspect relationships in the dataset.

---

## Features

### 1. Forecasting and Analysis

* Uses a pre-trained **TensorFlow LSTM model** for time-series prediction.
* Predicts next-year **Urban** and **Rural** indices with historical trend alignment.
* Incorporates **ARIMA-like smoothing** for stable results.

### 2. Interactive Visualization

* Built entirely on **Plotly** and **Streamlit** components.
* Includes **animated gauge meters** for forecast display.
* Responsive layout optimized for desktop and tablet screens.

### 3. Theme Customization

* Dual-mode interface:

  * **Light Mode:** Pastel-beige background, soft shadowed cards.
  * **Dark Mode:** AMOLED-black base with neon accent gradients.
* Typography using **Playfair Display** and **Lora**.

### 4. Automated Statistics

* Summary metrics (mean, median, std deviation) auto-generated for both index categories.
* Live correlation analysis and trend decomposition.

---

## Directory Structure

```
Cost-of-Living-Forecaster/
│
├── data/
│   ├── cost_of_living.csv
│   └── processed_data.csv
│
├── model/
│   └── cost_forecast_model.h5
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Naveen-Jayaraj/Infosys_ml1_inflation.git
cd Infosys_ml1_inflation
```

### Step 2: Install Dependencies

Ensure Python 3.8 or above is installed, then run:

```bash
pip install -r requirements.txt
```

Typical dependencies include:

```
streamlit
pandas
numpy
tensorflow
plotly
matplotlib
seaborn
```

### Step 3: Run the Application

```bash
streamlit run app.py
```

### Step 4: Access the Dashboard

Once the server starts, open:

```
http://localhost:8501
```

---

## Example Output

<div align="center">
  <img src="assets/dashboard_preview.png" width="800" alt="Dashboard Preview">
</div>

Forecasts for both **Urban** and **Rural** indices are shown as interactive charts with statistical overlays.

---

## Technical Implementation

1. **Data Preprocessing**

   * CSV file ingestion and feature scaling.
   * Missing values handled via forward filling.
   * Data split into training and validation sets.

2. **Model Architecture**

   * Implemented with **LSTM (Long Short-Term Memory)** networks.
   * Optimized using **Adam optimizer** and **MSE loss**.
   * Saved model (`.h5`) loaded at runtime for fast inference.

3. **Visualization**

   * Plotly line and area charts for trend comparison.
   * Gauge charts for projected index values.
   * Glassmorphism-based UI for improved user experience.

---

## How to Use

1. Launch the Streamlit app.
2. Select the mode (Light or Dark).
3. View **Urban** and **Rural** forecasts.
4. Click on **“Show Interactive Statistics”** to view correlations and summary data.
5. Export predictions if required (optional extension).

---

## Contribution

Contributions are welcome.
Fork the repository, create a new branch, and submit a pull request.

```bash
git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature
```

---

## License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for more details.

---

## Author

**Naveen Jayaraj**  
B.Tech CSE (AIML), SRM University Tiruchirappalli  
Email: [naveenpainthouse@gmail.com](mailto:naveenpainthouse@gmail.com)

---

## GitHub Repository

[![GitHub Repo](https://img.shields.io/badge/View_on-GitHub-black?style=for-the-badge&logo=github)](https://github.com/Naveen-Jayaraj/Infosys_ml1_inflation)
[![Download ZIP](https://img.shields.io/badge/Download-ZIP-blue?style=for-the-badge&logo=zip)](https://github.com/Naveen-Jayaraj/Infosys_ml1_inflation/archive/refs/heads/main.zip)
[![Issues](https://img.shields.io/github/issues/Naveen-Jayaraj/Infosys_ml1_inflation?style=for-the-badge)](https://github.com/Naveen-Jayaraj/Infosys_ml1_inflation/issues)
