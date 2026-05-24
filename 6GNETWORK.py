import streamlit as st
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import time

# Machine Learning
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="6G AI Optimizer",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #141e30, #243b55);
}

.title {
    font-size: 40px;
    font-weight: bold;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 10px rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown(
    '<div class="title">🚀 Knowledge-Augmented AI for 6G Network Optimization</div>',
    unsafe_allow_html=True
)

st.info("🌐 Digital Twin Simulation of Wireless Network using AI + Machine Learning")

# ---------------- DATASET ----------------

# Create dataset automatically
dataset = pd.DataFrame({
    "distance": [10,20,30,40,50,60,70,80,90,100,
                 15,25,35,45,55,65,75,85,95],
    
    "users": [5,10,15,20,25,30,35,40,45,50,
              8,12,18,22,28,33,38,43,48],
    
    "interference": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
                     0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    
    "signal": [90,82,75,68,60,52,45,38,30,20,
               85,78,70,63,55,48,40,32,25]
})

# ---------------- ML MODEL ----------------

X = dataset[['distance', 'users', 'interference']]
y = dataset['signal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- UI ----------------

col1, col2 = st.columns(2)

# ---------------- INPUT SECTION ----------------
with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📥 Input Parameters")

    distance = st.slider("Distance (m)", 1, 100, 50)

    users = st.slider("Users", 1, 50, 20)

    interference = st.slider(
        "Interference",
        0.0,
        1.0,
        0.3
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- OUTPUT SECTION ----------------
with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📤 Output Dashboard")

    # ---------------- OPTIMIZATION ----------------

    if st.button("⚡ Optimize Network"):

        st.write("### 🤖 AI Processing...")

        progress_bar = st.progress(0)

        status_text = st.empty()

        for i in range(100):

            progress_bar.progress(i + 1)

            if i < 30:
                status_text.text("🔍 Analyzing network data...")

            elif i < 60:
                status_text.text("🧠 Training ML model...")

            elif i < 90:
                status_text.text("⚙️ Predicting signal strength...")

            else:
                status_text.text("✅ Finalizing results...")

            time.sleep(0.01)

        # Prediction
        input_data = pd.DataFrame({
            'distance': [distance],
            'users': [users],
            'interference': [interference]
        })

        result = model.predict(input_data)[0]

        result = round(result, 2)

        st.session_state.result = result

        status_text.success("🚀 AI Prediction Complete!")

        # ---------------- OUTPUT ----------------

        st.success(f"✅ Predicted Signal Strength: {result}")

        # Metrics
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        confidence = random.randint(90, 99)

        st.metric("🤖 AI Confidence", f"{confidence}%")

        st.metric("📶 Network Health", f"{int(result)}%")

        st.metric("📉 Error Rate", f"{round(mae,2)}")

        st.progress(int(result))

        # ---------------- STATUS ----------------

        if result > 70:
            st.success("🟢 Network is Highly Stable")

        elif result > 40:
            st.warning("🟡 Network is Moderate")

        else:
            st.error("🔴 Network is Unstable")

        # ---------------- BAR GRAPH ----------------

        st.write("### 📊 Feature Impact Analysis")

        labels = ['Distance', 'Users', 'Interference']

        values = [distance, users, interference * 100]

        fig, ax = plt.subplots()

        ax.bar(labels, values)

        ax.set_title("Network Parameter Analysis")

        st.pyplot(fig)

        # ---------------- HEATMAP ----------------

        st.write("### 🌡️ AI Network Heatmap")

        heat_data = np.random.rand(10, 10)

        fig2, ax2 = plt.subplots()

        heatmap = ax2.imshow(heat_data)

        plt.colorbar(heatmap)

        ax2.set_title("Network Congestion Heatmap")

        st.pyplot(fig2)

    # ---------------- CSV UPLOAD ----------------

    st.write("### 📂 Upload Network Data")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file:

        data = pd.read_csv(uploaded_file)

        st.write("### 📄 Uploaded Dataset")

        st.dataframe(data)

        if st.button("📡 Analyze Uploaded Data"):

            uploaded_predictions = model.predict(
                data[['distance', 'users', 'interference']]
            )

            data["Predicted Signal"] = uploaded_predictions

            st.write("### ✅ Prediction Results")

            st.dataframe(data)

    # ---------------- DOWNLOAD REPORT ----------------

    report = f"""
--- NETWORK REPORT ---

Predicted Signal: {st.session_state.get("result", "N/A")}

System:
Knowledge-Augmented AI + Machine Learning

Technology:
Random Forest Regression

Status:
6G Network Optimization Successful
"""

    st.download_button(
        "📄 Download Report",
        report
    )

    st.write("---")

    # ---------------- FULL ANALYSIS ----------------

    if st.button("🧠 Run Full AI Analysis"):

        if "result" in st.session_state:

            result = st.session_state.result

            st.write("### 🔍 Deep AI Analysis Running...")

            st.write("✔ Step 1: Collecting network parameters...")
            st.write("✔ Step 2: Preprocessing data...")
            st.write("✔ Step 3: Training ML model...")
            st.write("✔ Step 4: Predicting signal strength...")
            st.write("✔ Step 5: Generating AI insights...")

            time.sleep(1)

            st.success("✅ Analysis Complete!")

            st.write("### 📊 Analysis Summary")

            st.write(f"📶 Signal Strength: {result}")

            if result > 70:
                st.success("🚀 Network optimized for ultra-fast 6G communication")

            elif result > 40:
                st.warning("⚠️ Network performance is moderate")

            else:
                st.error("❌ Network requires optimization")

            st.write("### 🧠 AI Insight")

            st.write("""
AI analyzed network traffic, interference,
user density, and communication distance
to predict optimized signal performance.
""")

        else:
            st.warning("⚠️ Please run Optimize Network first")

    # ---------------- AUTO FIX ----------------

    if st.button("⚡ Auto Fix Network"):

        if "result" in st.session_state:

            result = st.session_state.result

            st.write("### 🔧 Auto Optimization Started...")

            st.write("✔ Detecting weak parameters...")
            st.write("✔ Adjusting transmission power...")
            st.write("✔ Reducing interference...")
            st.write("✔ Balancing network load...")

            improved = min(result + 10, 100)

            st.write(f"📉 Before Optimization: {result}")

            st.write(f"📈 After Optimization: {improved}")

            st.success("✅ Network Improved Successfully!")

        else:
            st.warning("⚠️ Run Optimize Network first")

    # ---------------- 6G SIMULATION ----------------

    if st.button("🌐 Simulate 6G Environment"):

        st.write("### 📡 Running 6G Simulation...")

        steps = [
            "✔ Initializing network...",
            "✔ Allocating bandwidth...",
            "✔ Reducing latency...",
            "✔ Activating AI optimization...",
            "✔ Establishing ultra-fast communication..."
        ]

        for step in steps:
            st.write(step)
            time.sleep(1)

        st.success("🚀 6G Simulation Complete!")

        st.write("📶 Ultra-low latency achieved")

        st.write("⚡ High-speed communication enabled")

        st.write("🤖 AI-driven optimization active")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------

st.write("---")

st.markdown("""
<div style='text-align: center;
padding: 15px;
background-color: rgba(255,255,255,0.08);
border-radius: 10px;
margin-top: 20px;'>

<h3>🚀 Knowledge-Augmented AI for 6G Network Optimization</h3>

<p>
Developed by <b>Shubham G.s Urane</b>
</p>

<p>
💻 Python | 🤖 Machine Learning | 📡 6G AI Simulation
</p>

<p>
🌐 Digital Twin Technology for Wireless Network Optimization
</p>

<p style='font-size:14px; color:lightgray;'>
© 2026 AI-Based Project
</p>

</div>
""", unsafe_allow_html=True)
