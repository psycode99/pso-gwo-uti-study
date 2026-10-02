import streamlit as st
import pandas as pd
import joblib
import json

model    = joblib.load('model.pkl')
encoders = joblib.load('encoders.pkl')

with open('selected_features.json') as f:
    features = json.load(f)

with open('categorical_options.json') as f:
    categorical_options = json.load(f)

st.set_page_config(page_title='UTI Prediction Tool', layout='centered')
st.title('UTI Prediction Tool')
st.write('Enter the patient clinical values below to receive a real-time prediction.')
st.markdown('---')

user_input = {}
with st.form('prediction_form'):
    for feature in features:
        if feature in categorical_options:
            options  = categorical_options[feature]
            selected = st.selectbox(label=feature, options=options)
            le = encoders[feature]
            user_input[feature] = int(le.transform([str(selected)])[0])
        else:
            user_input[feature] = st.number_input(label=feature, value=0.0, format='%.4f')

    submitted = st.form_submit_button('Predict')

if submitted:
    input_df    = pd.DataFrame([user_input])[features]
    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown('---')
    if prediction == 1:
        st.error(f'⚠️  UTI Likely — Confidence: {probability * 100:.1f}%')
    else:
        st.success(f'✅  UTI Unlikely — Confidence: {(1 - probability) * 100:.1f}%')

    st.caption(
        'This tool is a decision-support aid only. '
        'It does not replace clinical judgment or laboratory confirmation.'
    )