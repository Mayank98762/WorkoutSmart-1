import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import os
import json
import datetime
from dotenv import load_dotenv
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import json
from geopy.geocoders import Nominatim
import streamlit.components.v1 as components
from geopy.distance import geodesic
from streamlit_option_menu import option_menu
import requests


# Load key from Streamlit Secrets or .env fallback
try:
    openrouter_key = st.secrets["OPENROUTER_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

def generate_with_openrouter(prompt):
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "HTTP-Referer": "https://your-app-name.streamlit.app/",  # Replace with your app URL
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistralai/mistral-7b-instruct",  # Or try gpt-3.5, meta-llama, etc.
        "messages": [
            {"role": "system", "content": "You are a helpful and motivating fitness coach."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error: {e}"

st.set_page_config(page_title="Smart Workout Dashboard", layout="wide")

# Updated UI Theme
st.markdown("""
    <style>
    /* Main background (right side) */
    .block-container {
        background-color: #fff700;
        color: #000000;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #000000;
        font-weight: 700;
        text-transform: uppercase;
    }

    /* Inputs */
    input, textarea, select {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #333333 !important;
        border-radius: 8px;
        padding: 0.6rem;
    }

    /* Buttons */
    .stButton > button {
        background-color: #00ff5f;
        color: #000000;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
        transition: 0.3s ease-in-out;
    }

    .stButton > button:hover {
        background-color: #00cc4c;
        transform: scale(1.03);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111111;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Tabs */
    .stTabs [role="tab"] {
        background-color: #1c1c1c;
        color: #00ff5f;
        border-radius: 6px;
        padding: 10px;
        margin-right: 6px;
        font-weight: bold;
        text-transform: uppercase;
    }

    .stTabs [aria-selected="true"] {
        background-color: #00ff5f;
        color: #000000;
    }

    /* Card style for workout logs or blocks */
    .card {
        background-color: #f5f500;
        border-left: 6px solid #00ff5f;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.4);
    }

    /* Metrics */
    .stMetric {
        background-color: #fff700 !important;
        padding: 0.5rem 1rem !important;
        border: 1px solid #00ff5f !important;
        border-radius: 10px !important;
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    selected_tab = option_menu(
        menu_title="Smart Workout Dashboard",
        options=["Analysis","Time Series Modelling","AI Insights", "Design Personalised Workout", "Workout Log","Real Time Workout Tracking"],
        icons=["clipboard-data","graph-up-arrow","robot","heart","calendar2-week","geo-alt-fill"],
        menu_icon="cast",
        default_index=0,
    )

load_dotenv()

# Load and Save Logs
LOG_FILE = "workout_log.json"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f)

with open(LOG_FILE, "r") as f:
    logs = json.load(f)

def save_log(entry):
    logs.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f)

# Sidebar Navigation

if selected_tab == "Analysis":
    st.header("\U0001F4CA Data Analysis")

    uploaded_file = st.file_uploader("Upload your workout CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Preview Data")
        st.write(df.head())

        columns = df.columns.tolist()
        x_axis = st.selectbox("Select X-axis", columns, key="xaxis")
        y_axis = st.selectbox("Select Y-axis", columns, key="yaxis")
        graph_type = st.selectbox("Select Graph Type", ["Line", "Scatter", "Bar", "Histogram", "Box"])
        legend_attr = st.selectbox("Select column for Legend (optional)", ["None"] + columns, key="legend")

        stat_mode = st.selectbox("Aggregate Method (where applicable)", ["Sum", "Mean", "Median", "Mode"])

        fig, ax = plt.subplots()

        if graph_type == "Line":
            if legend_attr != "None":
                for val in df[legend_attr].unique():
                    subset = df[df[legend_attr] == val]
                    ax.plot(subset[x_axis], subset[y_axis], label=str(val))
                ax.legend(title=legend_attr)
            else:
                ax.plot(df[x_axis], df[y_axis])

        elif graph_type == "Scatter":
            if legend_attr != "None":
                for val in df[legend_attr].unique():
                    subset = df[df[legend_attr] == val]
                    ax.scatter(subset[x_axis], subset[y_axis], label=str(val))
                ax.legend(title=legend_attr)
            else:
                ax.scatter(df[x_axis], df[y_axis])

        elif graph_type == "Bar":
            if df[x_axis].dtype == 'object' or df[x_axis].nunique() < 50:
                grouped = df.groupby(x_axis)[y_axis]
                if stat_mode == "Sum":
                    grouped = grouped.sum().reset_index()
                elif stat_mode == "Mean":
                    grouped = grouped.mean().reset_index()
                elif stat_mode == "Median":
                    grouped = grouped.median().reset_index()
                elif stat_mode == "Mode":
                    grouped = grouped.agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index()
                ax.bar(grouped[x_axis], grouped[y_axis])
                plt.xticks(rotation=45)
            else:
                st.error("Bar chart requires categorical X-axis (e.g., Workout Type).")

        elif graph_type == "Histogram":
            ax.hist(df[y_axis], bins=20)

        elif graph_type == "Box":
            ax.boxplot(df[y_axis])
            ax.set_xticklabels([y_axis])

        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"{graph_type} Plot of {y_axis} vs {x_axis}")
        st.pyplot(fig)

elif selected_tab == "Time Series Modelling":
    st.header("Time Series Weight Prediction")

    st.subheader("Enter Personal Details")
    age = st.number_input("Age", min_value=10, max_value=100, step=1)
    height_cm = st.number_input("Height (in cm)", min_value=100, max_value=250, step=1)
    weight_kg = st.number_input("Current Weight (in kg)", min_value=30, max_value=200, step=1)
    activity_level = st.selectbox("Activity Level", ["Sedentary", "Lightly active", "Moderately active", "Very active", "Super active"])

    if st.button("Predict Weight"):
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        activity_multipliers = {
            "Sedentary": 1.2,
            "Lightly active": 1.375,
            "Moderately active": 1.55,
            "Very active": 1.725,
            "Super active": 1.9
        }
        tdee = bmr * activity_multipliers[activity_level]
        deficit = 500
        calories_per_kg = 7700

        def predict_weight(days):
            weight_loss = (deficit * days) / calories_per_kg
            return round(weight_kg - weight_loss, 2)

        st.subheader("\U0001F4C9 Projected Weights")
        st.write(f"Weight after 1 month: **{predict_weight(30)} kg**")
        st.write(f"Weight after 2 months: **{predict_weight(60)} kg**")
        st.write(f"Weight after 6 months: **{predict_weight(180)} kg**")

        st.subheader("\U0001F372 Macro Nutrient Breakdown")
        protein = weight_kg * 2
        fat = weight_kg * 0.8
        carbs = (tdee - (protein * 4 + fat * 9)) / 4
        st.write(f"Protein: {protein:.1f}g, Fat: {fat:.1f}g, Carbs: {carbs:.1f}g")

    st.header("🏋️ Exercise Calorie Estimator")

    gender = st.selectbox("What is your gender?", ["Male", "Female"])
    mins = st.number_input("Duration (minutes)", min_value=10, step=5)
    heart_rate_option = st.selectbox("Do you know the average Heart Rate during the workout?(This will really help in making an accurate calorie expenditure calculation)", ["No", "Yes"])

    if heart_rate_option == 'No':
        exercise_type = st.selectbox("Choose exercise", ["Jogging", "Cycling", "Weight Lifting", "Swimming"])

        if exercise_type == "Jogging":
            met = 10

        elif exercise_type == "Cycling":
            met = 8

        elif exercise_type in ["Weight Lifting", "Swimming"]:
            intensity = st.selectbox("Please select the intensity of the Workout", ["Light Effort", "Moderate Effort", "Vigorous Effort"])
            if intensity == 'Light Effort':
                met = 3
            elif intensity == 'Moderate Effort':
                met = 4.5
            elif intensity == 'Vigorous Effort':
                met = 6
            else:
                met = 0  # fallback

        else:
            met = 0  # fallback

        calories_burned = met * weight_kg * mins / 60
        if calories_burned > 0:
            st.success(f"Calories Burned: {calories_burned:.2f}")
            st.subheader("\U0001F4DD Weekly Plan (Recommended)")
            st.markdown("- Mon/Wed/Fri: Full-body training\n- Tue/Thu: Cardio (45 mins)\n- Sat: Light Yoga or Stretch\n- Sun: Rest")
        else:
            st.warning("Please complete all selections to calculate calories.")

    elif heart_rate_option == 'Yes':
        bpm = st.number_input("How much was the average heart rate?")
        exercise_type = st.selectbox("Choose exercise", ["Jogging", "Cycling", "Weight Lifting", "Swimming"])

        if exercise_type in ["Jogging", "Cycling"]:
            if gender == 'Male':
                calories_burned = ((-55.0969 + (0.6309 * bpm) + (0.1988 * weight_kg) + (0.2017 * age)) / 4.184) * mins
            else:
                calories_burned = ((-20.4022 + (0.4472 * bpm) - (0.1263 * weight_kg) + (0.074 * age)) / 4.184) * mins

            if calories_burned > 0:
                st.success(f"Calories Burned: {calories_burned:.2f}")
                st.subheader("\U0001F4DD Weekly Plan (Recommended)")
                st.markdown("- Mon/Wed/Fri: Full-body training\n- Tue/Thu: Cardio (45 mins)\n- Sat: Light Yoga or Stretch\n- Sun: Rest")
            else:
                st.warning("Please enter realistic values for heart rate, age, and weight.")

        elif exercise_type == "Swimming":
            swim_type = st.selectbox("Please choose the swimming style:", [
                "Leisurely swimming", "Backstroke", "Breaststroke",
                "Freestyle (slow)", "Freestyle (moderate)", "Freestyle (fast)",
                "Butterfly", "Treading water (moderate)", "Treading water (vigorous)"
            ])

            met_dict = {
                "Leisurely swimming": 6,
                "Backstroke": 4.8,
                "Breaststroke": 5.3,
                "Freestyle (slow)": 5.8,
                "Freestyle (moderate)": 8.3,
                "Freestyle (fast)": 9.8,
               "Butterfly": 13.8,
                "Treading water (moderate)": 3.5,
                "Treading water (vigorous)": 7,
            }

            met = met_dict.get(swim_type, 0)
            calories = met * weight_kg * (mins / 60)

            if calories > 0:
                st.success(f"Calories Burned: {calories:.2f}")
                st.subheader("\U0001F4DD Weekly Plan (Recommended)")
                st.markdown("- Mon/Wed/Fri: Full-body training\n- Tue/Thu: Cardio (45 mins)\n- Sat: Light Yoga or Stretch\n- Sun: Rest")
            else:
                st.warning("Please select all values properly.")

        elif exercise_type == "Weight Lifting":
            intensity = st.selectbox("Please select the intensity of the Workout", ["Light Effort", "Moderate Effort", "Vigorous Effort"])
            if intensity == 'Light Effort':
                met = 3
            elif intensity == 'Moderate Effort':
                met = 4.5
            elif intensity == 'Vigorous Effort':
                met = 6
            else:
                met = 0

            calories_burned = met * weight_kg * mins / 60
            if calories_burned > 0:
                st.success(f"Calories Burned: {calories_burned:.2f}")
                st.subheader("\U0001F4DD Weekly Plan (Recommended)")
                st.markdown("- Mon/Wed/Fri: Full-body training\n- Tue/Thu: Cardio (45 mins)\n- Sat: Light Yoga or Stretch\n- Sun: Rest")
            else:
                st.warning("Please select all values properly.")

elif selected_tab == "AI Insights":
    st.header("\U0001F916 AI-Generated Fitness Insights")

    uploaded_file = st.file_uploader("Upload your workout dataset (CSV)", type=["csv"], key="aiupload")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(df.head())

        if st.button("Generate Insights"):
            st.subheader("\U0001F4AC CoachFit's Analysis")
            with st.spinner("Running CoachFit to generate insights..."):
                preview = df.head(5).to_string()
                summary = df.describe(include='all').to_string()
                correlation = df.corr(numeric_only=True).to_string()
                missing = df.isnull().sum().to_string()

                prompt = f"""
You are a fitness data analyst. Based on the uploaded workout dataset, generate AI-powered insights.

DATA PREVIEW:
{preview}

STATISTICAL SUMMARY:
{summary}

CORRELATION MATRIX:
{correlation}

MISSING VALUE REPORT:
{missing}

Instructions:
- Provide 3 key insights (patterns, trends, outliers, or anomalies)
- Provide 3 actionable recommendations to improve health/workout
- Suggest 2 charts/plots that would help visualize the trends

Format everything in clear bullet points.
"""
                output = generate_with_openrouter(prompt)
            st.markdown(output)

elif selected_tab == "Workout Log":
    st.header("\U0001F4CB Daily Workout Logger")
    today = datetime.date.today()
    st.subheader(f"Log your workout for {today}")

    duration = st.number_input("Duration (minutes)", min_value=10)
    calories = st.number_input("Calories Burned (optional)", value=0)

    st.subheader("Select Exercises Done")
    exercise_list = ["Jogging", "Weight Lifting", "Cycling", "Yoga"]
    selected_exercises = st.multiselect("Choose all that apply:", exercise_list)

    exercise_details = {}
    for exercise in selected_exercises:
        if exercise == "Jogging":
            distance = st.number_input("Distance jogged (in km)", key="jogging")
            exercise_details["Jogging"] = {"Distance (km)": distance}
        elif exercise == "Weight Lifting":
            sets = st.number_input("Number of sets", key="wl_sets")
            weight = st.number_input("Weight lifted per set (in kg)", key="wl_weight")
            exercise_details["Weight Lifting"] = {"Sets": sets, "Weight per set": weight}
        elif exercise == "Cycling":
            km = st.number_input("Distance cycled (km)", key="cycling")
            exercise_details["Cycling"] = {"Distance (km)": km}
        elif exercise == "Yoga":
            minutes = st.number_input("Minutes of yoga", key="yoga")
            exercise_details["Yoga"] = {"Minutes": minutes}

    if st.button("Log Workout"):
        entry = {
            "date": str(today),
            "duration": duration,
            "calories": calories,
            "exercises": exercise_details
        }
        save_log(entry)
        st.success("Workout logged successfully!")

    st.subheader("\U0001F4C5 Your Streak")
    log_dates = [datetime.datetime.strptime(log["date"], "%Y-%m-%d").date() for log in logs]
    log_dates = sorted(set(log_dates))
    streak = 1
    for i in range(len(log_dates)-1, 0, -1):
        if (log_dates[i] - log_dates[i-1]).days == 1:
            streak += 1
        else:
            break
    st.write(f"Current streak: **{streak}** days")
    if streak in [10, 20, 30]:
        st.balloons()
        st.success(f"Congrats on a {streak}-day streak! Keep going!")

    st.subheader("📜 Past Workout Logs")
    
    for entry in reversed(logs[-7:]):
        with st.expander(f"Workout on {entry['date']}"):
            st.write(f"Duration: {entry['duration']} minutes")
            st.write(f"Calories Burned: {entry['calories']} kcal")
            for ex_type, details in exercise_details.items():
                st.write(f"**{ex_type}**")
                for key, value in details.items():
                    st.write(f"- {key}: {value}")
    else:
        st.info("No logs found yet.")

elif selected_tab=='Design Personalised Workout':
    st.header("Use 🤖 AI for making a personalised Workout Plan for yourself")
    decision=st.selectbox("Are you trying to Loose Weight or Gain Weight?",["Loose Weight","Gain Weight"])
    current_weight = st.number_input("Current Weight (kg)?")
    aim = st.number_input("How many Kg's to {}?".format("lose" if decision == "Loose Weight" else "gain"))
    days = st.number_input("In how many days to achieve this?")
    exercise_hours = st.number_input("Time you can dedicate each day (minutes)?")
    gym = st.selectbox("Access to Gym?", ["Yes", "No"])
    days_per_week = st.slider("Workout days per week", 1, 7)  
    workout_type = st.selectbox("Preferred workout type", ["Bodyweight", "Gym Machines", "Free Weights", "Cardio-focused", "Mixed"])
    fitness_level = st.selectbox("Current fitness level", ["Beginner", "Intermediate", "Advanced"])
    injuries = st.text_input("Injuries/limitations to consider?")
    if st.button("Generate Workout Plan"):
        with st.spinner("Generating custom workout plan with CoachFit..."):
            prompt = f"""
You are a professional fitness coach. Based on the following goals and constraints, generate a **daily detailed workout plan** for the entire duration:

Goal: {decision} {aim} kg in {days} days
Current weight: {current_weight} kg
Daily Exercise Time: {exercise_hours} minutes
Days/Week: {days_per_week}
Workout Type: {workout_type}
Fitness Level: {fitness_level}
Gym Access: {gym}
Injuries or limitations: {injuries}

Instructions:
- Provide a complete workout schedule for {days} days
- Each day should include warm-up, main workout, cool-down
- Mention sets, reps, rest time
- Tailor intensity to fitness level
"""
            output = generate_with_openrouter(prompt)

            st.session_state['recommended_plan'] = output
            st.subheader("Your Personalized Workout Plan")
            st.markdown(output)

elif selected_tab=="Real Time Workout Tracking":
    st.header("🏃‍♂️ Live Workout Route Tracker")

    # Initialize session state
    if 'route_coords' not in st.session_state:
        st.session_state.route_coords = []
    if 'route_times' not in st.session_state:
        st.session_state.route_times = []

    st.info("Enable GPS/location access in your browser to track route in real-time.")

    # Inject JavaScript to capture GPS every 5 seconds
    components.html("""
        <script>
        const sendLocation = () => {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const coords = position.coords;
                    const msg = coords.latitude + "," + coords.longitude;
                    const streamlitEvent = new CustomEvent("streamlit:location", {
                        detail: msg
                    });
                    window.dispatchEvent(streamlitEvent);
                },
                (err) => console.log("Location access denied or unavailable.")
            );
        };
        setInterval(sendLocation, 5000); // every 5 seconds
        </script>
    """, height=0)

    # Hidden input to get JS data into Streamlit
    components.html("""
        <script>
        window.addEventListener("streamlit:location", (e) => {
            const coords = e.detail;
            const input = window.parent.document.querySelector('input[data-testid="stTextInput"]');
            if (input) {
                input.value = coords;
                input.dispatchEvent(new Event("input", { bubbles: true }));
            }
        });
        </script>
        <input type="text" style="opacity:0; height:0;" />
    """, height=0)

    coords = st.text_input("Live GPS (Hidden Field)", "")

    # Append coordinates and time
    if coords:
        try:
            lat, lon = map(float, coords.split(","))
            st.session_state.route_coords.append([lat, lon])
            st.session_state.route_times.append(datetime.now())
        except:
            pass

    # Show map
    if st.session_state.route_coords:
        m = folium.Map(location=st.session_state.route_coords[-1], zoom_start=16)
        folium.PolyLine(st.session_state.route_coords, color="blue", weight=4).add_to(m)
        folium.Marker(st.session_state.route_coords[-1], tooltip="📍 You").add_to(m)
        st_folium(m, width=700, height=500)

        # Distance calculation
        total_distance_km = 0.0
        coords_list = st.session_state.route_coords
        for i in range(1, len(coords_list)):
            total_distance_km += geodesic(coords_list[i-1], coords_list[i]).km
        st.success(f"📏 Distance Covered: {total_distance_km:.2f} km")

        # Save to CSV
        if st.button("💾 Save Route to CSV"):
            df = pd.DataFrame({
                'Timestamp': st.session_state.route_times,
                'Latitude': [c[0] for c in coords_list],
                'Longitude': [c[1] for c in coords_list]
            })
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Route CSV",
                data=csv,
                file_name='workout_route.csv',
                mime='text/csv'
            )

    # End workout
    if st.button("❌ End Workout & Clear Route"):
        st.session_state.route_coords = []
        st.session_state.route_times = []
        st.success("Workout route cleared.")