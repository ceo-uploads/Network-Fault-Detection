import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from data_utils import generate_network_data, FEATURE_COLUMNS, TARGET_COLUMN, validate_dataset

st.set_page_config(page_title="Network Fault Detection", page_icon="◈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#f4f6fa;color:#172033;}
.block-container {padding-top:1.7rem; padding-bottom:3rem; max-width:1450px;}
.hero {background:linear-gradient(115deg,#101d35,#18375a 65%,#20516c);padding:30px 34px;border-radius:18px;color:white;margin-bottom:22px;box-shadow:0 12px 30px rgba(15,35,65,.12);}
.hero h1 {font-family:Manrope,sans-serif;font-size:30px;font-weight:800;margin:0 0 8px;}
.hero p {color:#c6d5e7;margin:0;font-size:14px;}
.eyebrow {letter-spacing:2px;text-transform:uppercase;color:#85c5d8;font-size:10px;font-weight:700;margin-bottom:10px;}
.metric-card {background:white;border:1px solid #e7ebf2;border-radius:13px;padding:17px 18px;min-height:104px;box-shadow:0 3px 12px rgba(25,40,65,.035);}
.metric-label {font-size:11px;color:#758198;font-weight:700;text-transform:uppercase;letter-spacing:.8px;}
.metric-value {font-family:Manrope,sans-serif;font-size:25px;font-weight:800;color:#18263d;margin-top:8px;}
.metric-note {font-size:11px;color:#8994a6;margin-top:3px;}
.section-head {font-family:Manrope,sans-serif;font-size:19px;font-weight:800;color:#18263d;margin:12px 0 4px;}
.section-sub {font-size:12px;color:#7b879a;margin-bottom:14px;}
.panel {background:white;border:1px solid #e7ebf2;border-radius:14px;padding:18px 20px;}
div[data-testid="stSidebar"] {background:#101d35;}
div[data-testid="stSidebar"] * {color:#e7eef8;}
.stButton>button {border-radius:9px;font-weight:700;border:0;padding:.58rem 1rem;}
.stButton>button[kind="primary"] {background:#176b87;}
div[data-testid="stMetric"] {background:white;border:1px solid #e7ebf2;padding:14px;border-radius:12px;}
.stTabs [data-baseweb="tab"] {font-weight:700;}
div[data-testid="stDataFrame"] {border:1px solid #e7ebf2;border-radius:10px;}
.small-muted {color:#7b879a;font-size:12px;}
</style>
""", unsafe_allow_html=True)

MODEL_NAMES = ["Random Forest", "Support Vector Machine (SVM)", "Gradient Boosting"]

@st.cache_data
def default_data():
    return generate_network_data(2500, random_state=42)

def build_model(name):
    if name == "Random Forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=180, max_depth=14, min_samples_split=4,
                                              class_weight="balanced", random_state=42, n_jobs=-1))
        ])
    if name == "Support Vector Machine (SVM)":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(C=3.0, kernel="rbf", gamma="scale", probability=True,
                          class_weight="balanced", random_state=42))
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(n_estimators=120, learning_rate=0.08,
                                              max_depth=3, random_state=42))
    ])

def train_all(df, test_size):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    results = {}
    for name in MODEL_NAMES:
        model = build_model(name)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "model": model, "pred": pred, "X_test": X_test, "y_test": y_test,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
            "report": classification_report(y_test, pred, output_dict=True, zero_division=0)
        }
    return results

if "dataset" not in st.session_state:
    st.session_state.dataset = default_data()
if "results" not in st.session_state:
    st.session_state.results = None
if "trained_on" not in st.session_state:
    st.session_state.trained_on = None

with st.sidebar:
    st.markdown("## ◈ NETSCOPE")
    st.caption("NETWORK INTELLIGENCE")
    st.markdown("---")
    page = st.radio("WORKSPACE", ["Overview", "Dataset Explorer", "Model Laboratory", "Fault Detection", "Evaluation"], label_visibility="visible")
    st.markdown("---")
    st.markdown("**PROJECT**")
    st.caption("Network Fault Detection")
    st.caption("Supervised Classification")
    st.markdown("---")
    st.caption("Academic Data Mining Project")

df = st.session_state.dataset
class_count = df[TARGET_COLUMN].nunique()
st.markdown("""
<div class="hero">
  <div class="eyebrow">NETWORK INTELLIGENCE / DATA MINING</div>
  <h1>Network Fault Detection</h1>
  <p>Analyze network telemetry, train classification models and identify network faults from performance metrics.</p>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    st.markdown('<div class="section-head">System overview</div><div class="section-sub">A unified workspace for network telemetry analysis and fault classification.</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col, label, value, note in [
        (c1,"DATA RECORDS",f"{len(df):,}","Current dataset size"),
        (c2,"INPUT FEATURES",str(len(FEATURE_COLUMNS)),"Network telemetry fields"),
        (c3,"FAULT CLASSES",str(class_count),"Classification categories"),
        (c4,"TRAINED MODELS",str(3 if st.session_state.results else 0),"Available in model lab")
    ]:
        col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)
    st.write("")
    left,right = st.columns([1.15, .85])
    with left:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown("**Fault distribution**")
        counts = df[TARGET_COLUMN].value_counts().rename_axis("Network condition").reset_index(name="Records")
        fig = px.bar(counts, x="Network condition", y="Records", color="Network condition",
                     color_discrete_sequence=["#176b87","#e0a23a","#c85b61","#5a78b8","#4a9b83"])
        fig.update_layout(showlegend=False, height=320, margin=dict(l=10,r=10,t=15,b=10), paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown("**Detection workflow**")
        for n,t,d in [("01","Telemetry input","Upload or use the sample dataset."),("02","Model training","Train three classification algorithms."),("03","Model evaluation","Compare accuracy, precision, recall and F1."),("04","Fault prediction","Enter metrics and identify the likely condition.")]:
            st.markdown(f"**{n}  {t}**  \n<span class='small-muted'>{d}</span>",unsafe_allow_html=True)
            st.write("")
        st.markdown('</div>',unsafe_allow_html=True)
    st.write("")
    st.info("Start in Model Laboratory and train all three models before using the Fault Detection workspace.")

elif page == "Dataset Explorer":
    st.markdown('<div class="section-head">Dataset explorer</div><div class="section-sub">Inspect network telemetry and optionally load your own labeled CSV dataset.</div>',unsafe_allow_html=True)
    with st.expander("Upload a dataset", expanded=False):
        uploaded = st.file_uploader("CSV file", type=["csv"], help="Required columns: the five network features and fault_type.")
        if uploaded is not None:
            try:
                candidate = pd.read_csv(uploaded)
                valid, message = validate_dataset(candidate)
                if valid:
                    st.session_state.dataset = candidate
                    st.session_state.results = None
                    st.success("Dataset loaded. Train the models again to evaluate this dataset.")
                    st.rerun()
                else:
                    st.error(message)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
    st.markdown("**Dataset preview**")
    st.dataframe(df.head(100),use_container_width=True,hide_index=True)
    a,b = st.columns(2)
    with a:
        st.markdown("**Feature summary**")
        st.dataframe(df[FEATURE_COLUMNS].describe().T,use_container_width=True)
    with b:
        st.markdown("**Class distribution**")
        st.dataframe(df[TARGET_COLUMN].value_counts().rename("records").to_frame(),use_container_width=True)
    st.caption("CSV schema: latency_ms, packet_loss_pct, bandwidth_mbps, cpu_usage_pct, connections, fault_type.")

elif page == "Model Laboratory":
    st.markdown('<div class="section-head">Model laboratory</div><div class="section-sub">Train and compare three classification algorithms using the same stratified train/test split.</div>',unsafe_allow_html=True)
    with st.container(border=True):
        c1,c2 = st.columns([1,2])
        with c1:
            test_percent = st.slider("Test dataset (%)", 15, 35, 20, 5)
        with c2:
            st.markdown("**Algorithms included**")
            st.caption("Random Forest · Support Vector Machine · Gradient Boosting")
        if st.button("Train all three models",type="primary",use_container_width=True):
            with st.spinner("Training models and calculating evaluation metrics..."):
                try:
                    st.session_state.results = train_all(df, test_percent/100)
                    st.session_state.trained_on = len(df)
                    st.success("All three models trained successfully.")
                except Exception as e:
                    st.error(f"Training failed: {e}")
    if st.session_state.results:
        rows=[]
        for name,r in st.session_state.results.items():
            rows.append({"Model":name,"Accuracy":r["accuracy"],"Precision":r["precision"],"Recall":r["recall"],"F1-Score":r["f1"]})
        metrics=pd.DataFrame(rows).set_index("Model")
        st.markdown("**Performance comparison**")
        st.dataframe(metrics.style.format("{:.3f}"),use_container_width=True)
        plotdf=metrics.reset_index().melt(id_vars="Model",var_name="Metric",value_name="Score")
        fig=px.bar(plotdf,x="Model",y="Score",color="Metric",barmode="group",range_y=[0,1.05],
                   color_discrete_sequence=["#176b87","#e0a23a","#5a78b8","#4a9b83"])
        fig.update_layout(height=370,plot_bgcolor="white",paper_bgcolor="white",margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig,use_container_width=True)
        best=max(st.session_state.results,key=lambda k:st.session_state.results[k]["f1"])
        st.success(f"Best model by weighted F1-score: {best}")
    else:
        st.info("No trained models yet. Select the test size and click Train all three models.")

elif page == "Fault Detection":
    st.markdown('<div class="section-head">Live fault detection</div><div class="section-sub">Enter network telemetry values, choose a trained model and generate a prediction.</div>',unsafe_allow_html=True)
    if not st.session_state.results:
        st.warning("Train the three models first in Model Laboratory.")
    else:
        selected=st.selectbox("Select classification model",MODEL_NAMES)
        st.markdown("**Network telemetry input**")
        with st.form("prediction_form"):
            c1,c2,c3=st.columns(3)
            latency=c1.number_input("Latency (ms)",0.0,5000.0,35.0,step=1.0)
            loss=c2.number_input("Packet loss (%)",0.0,100.0,0.5,step=0.1)
            bandwidth=c3.number_input("Bandwidth (Mbps)",0.0,100000.0,500.0,step=10.0)
            cpu=c1.number_input("CPU usage (%)",0.0,100.0,35.0,step=1.0)
            connections=c2.number_input("Active connections",0,1000000,120,step=10)
            submitted=st.form_submit_button("Analyze network condition",type="primary",use_container_width=True)
        if submitted:
            row=pd.DataFrame([[latency,loss,bandwidth,cpu,connections]],columns=FEATURE_COLUMNS)
            model=st.session_state.results[selected]["model"]
            prediction=model.predict(row)[0]
            probabilities=model.predict_proba(row)[0]
            classes=model.named_steps["model"].classes_
            confidence=float(np.max(probabilities))*100
            st.markdown("---")
            st.markdown("**Detection result**")
            color="#18794e" if prediction=="Normal" else "#b45309"
            st.markdown(f'<div class="panel"><div class="metric-label">PREDICTED NETWORK CONDITION</div><div style="font-family:Manrope;font-size:30px;font-weight:800;color:{color};margin:8px 0">{prediction}</div><div class="small-muted">Model: {selected} &nbsp;·&nbsp; Confidence: {confidence:.1f}%</div></div>',unsafe_allow_html=True)
            probdf=pd.DataFrame({"Condition":classes,"Probability":probabilities}).sort_values("Probability",ascending=False)
            fig=px.bar(probdf,x="Probability",y="Condition",orientation="h",range_x=[0,1],text=probdf["Probability"].map(lambda x:f"{x:.1%}"))
            fig.update_traces(marker_color="#176b87",textposition="outside")
            fig.update_layout(height=280,plot_bgcolor="white",paper_bgcolor="white",margin=dict(l=10,r=35,t=10,b=10))
            st.plotly_chart(fig,use_container_width=True)
            if prediction=="Normal":
                st.success("The selected model classifies the supplied metrics as Normal.")
            else:
                st.warning(f"The selected model indicates a possible {prediction} condition. Verify against live network monitoring tools.")
            st.caption("Prediction confidence is the model's estimated class probability, not a guarantee of real-world fault severity.")

elif page == "Evaluation":
    st.markdown('<div class="section-head">Model evaluation</div><div class="section-sub">Inspect confusion matrices and per-class classification results for each trained model.</div>',unsafe_allow_html=True)
    if not st.session_state.results:
        st.warning("Train the models first in Model Laboratory.")
    else:
        selected=st.selectbox("Choose model to inspect",MODEL_NAMES,key="eval_model")
        result=st.session_state.results[selected]
        labels=sorted(result["y_test"].unique())
        cm=confusion_matrix(result["y_test"],result["pred"],labels=labels)
        left,right=st.columns([1,1])
        with left:
            st.markdown("**Confusion matrix**")
            fig=px.imshow(cm,x=labels,y=labels,text_auto=True,color_continuous_scale="Blues",labels=dict(x="Predicted",y="Actual",color="Records"))
            fig.update_layout(height=400,margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig,use_container_width=True)
        with right:
            st.markdown("**Overall metrics**")
            for label,key in [("Accuracy","accuracy"),("Precision","precision"),("Recall","recall"),("F1-score","f1")]:
                st.metric(label,f"{result[key]:.3f}")
        st.markdown("**Classification report**")
        report=pd.DataFrame(result["report"]).T
        st.dataframe(report.style.format("{:.3f}",na_rep="-"),use_container_width=True)
        st.download_button("Download classification report (CSV)",report.to_csv().encode("utf-8"),file_name=f"{selected.lower().replace(' ','_')}_report.csv",mime="text/csv")

st.markdown("---")
st.caption("NETWORK FAULT DETECTION  /  DATA MINING COURSE PROJECT")
