import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from io import StringIO
from PIL import Image
import plotly.express as px

# =========================
# PAGE CONFIG & STYLING
# =========================
st.set_page_config(page_title="ADMET & Docking Prioritizer", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            color: #fafafa;
        }
        h1, h2, h3 {
            color: #00c3ff;
        }
        div.stButton > button {
            background-color: #0078d7;
            color: white;
            border-radius: 8px;
            height: 3em;
            font-weight: 600;
        }
        div.stButton > button:hover {
            background-color: #009eff;
            color: white;
        }
        .stAlert {
            background-color: #1e2130;
            border-left: 0.25rem solid #00c3ff;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 ADMET & Docking Prioritizer App")
st.caption("Integrated **Medicinal Chemistry**, **Cheminformatics**, and **Docking** for candidate prioritization.")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/88/Molecular_structure_example.svg", use_container_width=True)
    st.markdown("### ⚙️ Quick Guide")
    st.markdown("""
    1. Choose **input method**
    2. Click **Analyze**
    3. View **Drug-likeness, ADMET, and Docking rank**
    """)
    st.markdown("---")
    st.markdown("🧪 Built with **RDKit + Streamlit + Plotly**")

# =========================
# EXAMPLE DATA
# =========================
EXAMPLE_DATA = """SMILES,Docking_Score
CC(=O)Oc1ccccc1C(=O)O,-7.2
COc1ccc(C(C)Nc2ccc(C)cc2)cc1,-6.5
O=C1CCc2c(C)nc(C)c2N1C1CC1,-4.1
CC(C)CN(C)CC(O)C(C)c1ccc(O)c(Cl)c1,-5.9
C1=CC=C2C(=C1)C=CC(=O)C2=O,-7.5
InvalidSMILES,-8.0
"""

# =========================
# INPUT SELECTION
# =========================
st.header("🧾 Input Candidate Molecules")

input_method = st.radio(
    "Select Input Method:",
    ('Use Example Data', 'Paste Custom Data', 'Upload CSV File'),
    horizontal=True
)

if input_method == 'Use Example Data':
    raw_data = EXAMPLE_DATA
    df_input = pd.read_csv(StringIO(raw_data.strip()))

elif input_method == 'Paste Custom Data':
    raw_data = st.text_area(
        "Paste your data (columns: SMILES, Docking_Score):",
        value=EXAMPLE_DATA,
        height=200
    )
    df_input = pd.read_csv(StringIO(raw_data.strip()))

else:
    uploaded_file = st.file_uploader("Upload CSV file (SMILES, Docking_Score)", type=["csv"])
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = None

# =========================
# ANALYSIS PIPELINE
# =========================
if st.button("🚀 Analyze & Prioritize Candidates") and df_input is not None:
    try:
        if 'SMILES' not in df_input.columns or 'Docking_Score' not in df_input.columns:
            st.error("❌ Input data must contain 'SMILES' and 'Docking_Score' columns.")
        else:
            with st.spinner("⏳ Computing molecular descriptors and ADMET properties..."):
                results = []
                images = []
                for _, row in df_input.iterrows():
                    smiles = row['SMILES']
                    docking_score = row['Docking_Score']
                    mol = Chem.MolFromSmiles(smiles)

                    if mol is None:
                        results.append({
                            'SMILES': smiles,
                            'Docking_Score': docking_score,
                            'Status': 'Invalid SMILES',
                            'MW': None, 'LogP': None, 'HDonors': None, 'HAcceptors': None,
                            'Violations': None, 'ADMET_Predict': None
                        })
                        images.append(None)
                        continue

                    # Compute basic properties
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    h_donors = Descriptors.NumHDonors(mol)
                    h_acceptors = Descriptors.NumHAcceptors(mol)

                    # Lipinski violations
                    violations = sum([
                        mw > 500,
                        logp > 5,
                        h_donors > 5,
                        h_acceptors > 10
                    ])

                    # Mock ADMET Prediction (can later link to SwissADME/pkCSM)
                    admet_prediction = "Good" if (logp < 5 and h_donors <= 5) else "Moderate"

                    is_drug_like = violations <= 1
                    status = 'Pass' if is_drug_like else 'Fail (Lipinski Violation)'

                    results.append({
                        'SMILES': smiles,
                        'Docking_Score': docking_score,
                        'Status': status,
                        'MW': round(mw, 2),
                        'LogP': round(logp, 2),
                        'HDonors': h_donors,
                        'HAcceptors': h_acceptors,
                        'Violations': violations,
                        'ADMET_Predict': admet_prediction
                    })

                    # Store molecule image
                    images.append(Draw.MolToImage(mol, size=(200, 200)))

                df_results = pd.DataFrame(results)

                # Rank by Docking Score
                df_pass = df_results[df_results['Status'] == 'Pass'].copy()
                df_fail = df_results[df_results['Status'] != 'Pass'].copy()
                df_pass.sort_values(by='Docking_Score', ascending=True, inplace=True)
                df_pass['Final_Rank'] = range(1, len(df_pass) + 1)
                df_fail['Final_Rank'] = '-'
                df_final = pd.concat([df_pass, df_fail]).reset_index(drop=True)

            st.success("✅ Analysis Complete!")

            # =========================
            # RESULTS TABLE
            # =========================
            st.subheader("📊 Prioritized Drug Candidates")
            st.caption("Only 'Pass' molecules are ranked (lower Docking Score = better binding).")
            st.dataframe(df_final, use_container_width=True, hide_index=True)

            # =========================
            # STRUCTURE VISUALIZATION
            # =========================
            st.subheader("🧪 Molecular Structures")
            for i, mol_img in enumerate(images):
                if mol_img is not None:
                    st.image(mol_img, caption=f"{df_final.iloc[i]['SMILES']} | Score: {df_final.iloc[i]['Docking_Score']}", width=150)

            # =========================
            # INTERACTIVE CHARTS
            # =========================
            st.subheader("📈 Visual Analysis")
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.scatter(
                    df_final[df_final["Status"] == "Pass"],
                    x="MW", y="Docking_Score", color="Violations",
                    hover_data=["SMILES"], title="MW vs Docking Score"
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = px.bar(
                    df_final, x="Status", y="Violations", color="Status",
                    title="Lipinski Violations by Category"
                )
                st.plotly_chart(fig2, use_container_width=True)

            # =========================
            # SUMMARY METRICS
            # =========================
            pass_count = (df_final['Status'] == 'Pass').sum()
            fail_count = (df_final['Status'].str.contains('Fail')).sum()
            col1, col2 = st.columns(2)
            col1.metric("✅ Passed (Drug-like)", pass_count)
            col2.metric("❌ Failed (Violations)", fail_count)

            # =========================
            # DOWNLOAD RESULTS
            # =========================
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Results as CSV", csv, "admet_docking_results.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error during processing: {e}")

else:
    st.info("👆 Upload, paste, or load example data, then click **Analyze & Prioritize Candidates** to start.")

