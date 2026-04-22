import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from lifelines import WeibullAFTFitter
import shap
import matplotlib.pyplot as plt

# PAGE CONFIG
st.set_page_config(
    page_title="Brain Tumour Survival Dashboard",
    layout="wide"
)

st.title("Brain Tumour Survival Analysis Dashboard")

# LOAD MODELS
with open("model.pkl", "rb") as f:
    aft_model = pickle.load(f)

with open("train_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

with open("kmeans_pipeline.pkl", "rb") as f:
    kmeans_pipeline = pickle.load(f)

with open("age_scaler.pkl", "rb") as f:
    age_scaler = pickle.load(f)

# MODEL COLUMNS
MODEL_COLUMNS = [
    "Age",
    "Chemotherapy",
    "DiagnosisMethod_2",
    "DiagnosisMethod_11",
    "DiagnosisMethod_6",
    "DiagnosisMethod_7",
    "FamilyHistory",
    "Grade_1.0",
    "Grade_2.0",
    "Grade_3.0",
    "Grade_4.0",
    "Grade_9.0",
    "I10_C710",
    "I10_C711",
    "I10_C712",
    "I10_C713",
    "I10_C714",
    "I10_C716",
    "I10_C717",
    "I10_C718",
    "I10_C719",
    "I10_C111",
    "Laterality_11",
    "Laterality_2",
    "Laterality_3",
    "Laterality_9",
    "Morphology_Group_Carcinoma",
    "Morphology_Group_Ependymoma",
    "Morphology_Group_Glioblastoma",
    "Morphology_Group_Glioma",
    "Morphology_Group_Medulloblastoma",
    "Morphology_Group_Neoplasm",
    "Morphology_Group_Oligodendroglioma",
    "Morphology_Group_Astrocytoma",
    "Morphology_Group_Other",
    "OtherTreatment_1",
    "OtherTreatment_2",
    "OtherTreatment_3",
    "OtherTreatment_4",
    "Radiotherapy",
    "Sex",
    "Surgery",
    "Topology_710",
    "Topology_711",
    "Topology_712",
    "Topology_713",
    "Topology_714",
    "Topology_716",
    "Topology_717",
    "Topology_718",
    "Topology_719"
]

# PATIENT INFO
top_info1, top_info2 = st.columns([2, 1])

with top_info1:
    patient_name = st.text_input("Patient Name")

with top_info2:
    patient_id = st.text_input("Patient ID")

st.sidebar.header("Patient Inputs")

if st.sidebar.button("Reset Inputs"):
    st.session_state.clear()
    st.rerun()

# FEATURE VECTOR
input_vector = {col: 0 for col in MODEL_COLUMNS}

input_vector["Age"] = st.sidebar.slider("Age", 20, 100, 20)
input_vector["Age"] = age_scaler.transform([[input_vector["Age"]]])[0][0]

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
input_vector["Sex"] = 0 if sex == "Male" else 1

topology_map = {
    "C71.0 - Cerebrum": 710,
    "C71.1 - Frontal Lobe": 711,
    "C71.2 - Temporal Lobe": 712,
    "C71.3 - Parietal Lobe": 713,
    "C71.4 - Occipital Lobe": 714,
    "C71.6 - Cerebellum, NOS": 716,
    "C71.7 - Brain Stem": 717,
    "C71.8 - Overl. Lesion of Brain": 718,
    "C71.9 - Brain, NOS": 719,
    "Other": 111
}

topo = st.sidebar.selectbox("Topology", list(topology_map.keys()))
input_vector[f"Topology_{topology_map[topo]}"] = 1

morph_map = {
    "Carcinoma": "Carcinoma",
    "Ependymoma": "Ependymoma",
    "Glioblastoma": "Glioblastoma",
    "Glioma": "Glioma",
    "Medulloblastoma": "Medulloblastoma",
    "Neoplasm": "Neoplasm",
    "Oligodendroglioma": "Oligodendroglioma",
    "Astrocytoma":"Astrocytoma",
    "Other": "Other",
}

morph = st.sidebar.selectbox("Morphology", list(morph_map.keys()))
input_vector[f"Morphology_Group_{morph_map[morph]}"] = 1

lat_map = {
    "Right":2,
    "Left": 3,
    "Midline/Unknown": 9,
    "Other": 11
}

lat = st.sidebar.selectbox("Laterality", list(lat_map.keys()))
input_vector[f"Laterality_{lat_map[lat]}"] = 1

grade_map = {
    "I - Well Differenciated": 1.0,
    "II - Moderately Differenciated": 2.0,
    "III - Poorly Differenciated": 3.0,
    "IV - Undifferenciated/Anaplastic": 4.0,
    "Other/ Unknown": 9.0
}

grade = st.sidebar.selectbox("Grade", list(grade_map.keys()))
input_vector[f"Grade_{grade_map[grade]}"] = 1

I10_map = {
    "C71.0 - Malignant Neoplasm, Cerebrum, Except Lobes and Ventricles":"C710",
    "C71.1 - Malignant Neoplasm, Frontal Lobe": "C711",
    "C71.2 - Malignant Neoplasm, Temporal Lobe": "C712",
    "C71.3 - Malignant Neoplasm, Parietal Lobe": "C713",
    "C71.4 - Malignant Neoplasm, Occipital Lobe": "C714",
    "C71.6 - Malignant Neoplasm, Cerebellum": "C716",
    "C71.7 - Malignant Neoplasm, Brain Stem": "C717",
    "C71.8 - Malignant Neoplasm, Overlapping Sites of Brain": "C718",
    "C71.9 - Malignant Neoplasm of Brain, Unspecified": "C719",
    "Other": "C111"
}

i10 = st.sidebar.selectbox("I10", list(I10_map.keys()))
input_vector[f"I10_{I10_map[i10]}"] = 1

diag_map = {
    "Clinical Investigation/Ultra Sound": 2,
    "Other/ Unknown": 11,
    "Histology of Metastases": 6,
    "Histology of Primary": 7
}

diag = st.sidebar.selectbox("Diagnosis Method", list(diag_map.keys()))
input_vector[f"DiagnosisMethod_{diag_map[diag]}"] = 1

input_vector["Surgery"] = int(st.sidebar.checkbox("Surgery"))

input_vector["Radiotherapy"] = int(st.sidebar.checkbox("Radiotherapy"))

input_vector["Chemotherapy"] = int(st.sidebar.checkbox("Chemotherapy"))

input_vector["FamilyHistory"] = int(st.sidebar.checkbox("Family History"))

other_tx_map = {
    "Immunotherapy": 1,
    "Hormone Therapy": 2,
    "No Treatment": 3,
    "Other": 4
}

st.sidebar.write("Other Treatment")

selected = []

for label in other_tx_map:
    if st.sidebar.checkbox(label):
        selected.append(label)


if len(selected) > 1:
    st.sidebar.error("Please select only one other treatment option")
elif len(selected) == 1:
    other_tx_value = other_tx_map[selected[0]]
    input_vector[f"OtherTreatment_{other_tx_value}"] = 1
else:
    # No selection → all remain 0 (baseline)
    pass

final_df = pd.DataFrame([input_vector]).reindex(
    columns=feature_columns,
    fill_value=0
)

# PREDICT
if st.button("Predict"):
    
    final_df_model = final_df.copy()
    final_df_model = final_df.reindex(columns=feature_columns, fill_value=0)

        # SURVIVAL MODEL
    median_survival = aft_model.predict_median(final_df_model).iloc[0]
    expected_survival = aft_model.predict_expectation(final_df_model).iloc[0]
    surv_func = aft_model.predict_survival_function(final_df_model)

    time_points = {"6 Months": 6, "1 Year": 12, "3 Years": 36}
    survival_probs = {
        k: float(np.interp(t, surv_func.index, surv_func.iloc[:, 0]))
        for k, t in time_points.items()
    }

    risk_pct = 100 * (1 - survival_probs["3 Years"])
    risk_group = (
        "High Risk" if risk_pct > 70
        else "Medium Risk" if risk_pct > 30
        else "Low Risk"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Median Survival", f"{median_survival:.1f} months")

    with col2:
        st.metric("Expected Survival", f"{expected_survival:.1f} months")

    with col3:
        st.metric(
            "Risk Group & Score",
            f"{risk_group} ({risk_pct:.1f}%)"
            )

    # Survival probabilities row
    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("6-Month Survival", f"{survival_probs['6 Months']*100:.1f}%")

    with col5:
        st.metric("1-Year Survival", f"{survival_probs['1 Year']*100:.1f}%")

    with col6:
        st.metric("3-Year Survival", f"{survival_probs['3 Years']*100:.1f}%")

    st.markdown("---")

    cluster_features = {
    "Topology": topology_map[topo],
    "Morphology_Group": morph_map[morph],
    "DiagnosisMethod": diag_map[diag],
    "I10": I10_map[i10],
    "Laterality": lat_map[lat],
    "Grade": grade_map[grade],
    "OtherTreatment": other_tx_map[selected[0]] if selected else 0,

    "Chemotherapy": input_vector["Chemotherapy"],
    "FamilyHistory": input_vector["FamilyHistory"],
    "Radiotherapy": input_vector["Radiotherapy"],
    "Surgery": input_vector["Surgery"],
    "Sex": input_vector["Sex"],
    "Age": input_vector["Age"]
    }

    cluster_input = pd.DataFrame([cluster_features])

    patient_cluster = kmeans_pipeline.predict(cluster_input)[0]
    cluster_info = {
        0: ("Older Unspecified Tumor", [
            "Higher-than-average patient age",
            "Predominance of C71.9-coded tumors across topology and ICD classification",
            "Clinically consistent but anatomically non-specific tumor grouping"
            ]),

        1: ("Parietal Tumor", [
            "Strong dominance of C71.3 (parietal lobe tumors)",
            "Consistent alignment between topology and ICD coding",
            "Low occurrence of unspecified tumor classification"
            ]),

        2: ("Frontal Lobe Tumor", [
            "Strong dominance of C71.1 (frontal lobe tumors)",
            "Clear anatomical localization of tumor site",
            "Reduced presence of unspecified tumor cases"
            ]),

        3: ("Older Mixed Tumor", [
            "Moderately higher patient age",
            "Lower prevalence of C71.9-coded tumors",
            "More heterogeneous tumor distribution patterns"
            ]),

        4: ("Younger Low-Grade Tumor", [
            "Lower-than-average patient age",
            "Reduced occurrence of glioblastoma",
            "Potentially less aggressive tumor biology"
            ])
    }

    cluster_name, cluster_desc = cluster_info.get(
            patient_cluster,
            (f"Cluster {patient_cluster}", ["No interpretation available"])
        )
        
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Patient Archetype")
        st.write(f"**Cluster:** {cluster_name}")

        st.markdown("### Interpretation")

        for d in cluster_desc:
            st.write(f"- {d}")
    
    with right:
        st.subheader("Survival Curve")
        st.caption("Estimated probability of survival over time (0–36 months) based on the patient-specific survival model.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=surv_func.index,
            y=surv_func.iloc[:, 0],
            mode="lines",
            name="Survival"
        ))

        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(range=[0,1])
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Impact")
    st.caption("Quantifies the contribution of each patient characteristics to the deviation from baseline survival prediction."" Blue bars indicate features pushing predicted survival up (longer expected survival)."" Red bars indicate features pushing predicted survival down (shorter expected survival).")
        
    def predict_fn(X):
        X_df = pd.DataFrame(X, columns=feature_columns)
        return aft_model.predict_expectation(X_df).to_numpy()

    background = pd.DataFrame(
        np.zeros((50, len(feature_columns))),
        columns=feature_columns
        )

    explainer = shap.Explainer(predict_fn, background)
    shap_values = explainer(final_df_model)

    shap_vals = shap_values.values[0]
    features = shap_values.feature_names

    shap_df = pd.DataFrame({
        "Feature": features,
        "Impact": shap_vals
    })

    shap_df["abs"] = shap_df["Impact"].abs()
    shap_df = shap_df.sort_values("abs", ascending=False).head(10)

    fig_shap = px.bar(
        shap_df[::-1],
        x="Impact",
        y="Feature",
        orientation="h",
        color="Impact",
        color_continuous_scale=["orange", "white", "green"]
    )

    fig_shap.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10)
        )

    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("---")

    st.subheader("Clinical Summary")

    st.write(f"""
    **Patient:** {patient_name}  
    **ID:** {patient_id} 

    Patient **{patient_name}** belongs to **{cluster_name}**
    
    **Risk Group:** {risk_group}  
    **Risk Score:** {risk_pct:.1f}%

    **Survival Outcomes:**
    - 6 months: **{survival_probs['6 Months']*100:.1f}%**
    - 1 year: **{survival_probs['1 Year']*100:.1f}%**
    - 3 years: **{survival_probs['3 Years']*100:.1f}%**

    
    Model combines Weibull AFT survival regression and clustering to estimate prognosis.
    SHAP values show how each clinical feature shifts survival prediction for this patient.
    """)