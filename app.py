import streamlit as st
import pandas as pd
import base64
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, cohen_kappa_score

# Calculates performance matrix
def calc_metrics(input_data):
    Y_actual = input_data.iloc[:,0]
    Y_predicted = input_data.iloc[:,1]
    acc = accuracy_score(Y_actual, Y_predicted)
    balanced_acc = balanced_accuracy_score(Y_actual, Y_predicted)
    precision = precision_score(Y_actual, Y_predicted, average='weighted')
    recall = recall_score(Y_actual, Y_predicted, average='weighted')
    mcc = matthews_corrcoef(Y_actual, Y_predicted)
    f1 = f1_score(Y_actual, Y_predicted, average='weighted')
    cohen_kappa = cohen_kappa_score(Y_actual, Y_predicted)

    acc_series = pd.Series(acc, name='Accuracy')
    balanced_series = pd.Series(balanced_acc, name='Balanced Accuracy')
    precision_series = pd.Series(precision, name='Precision')
    recall_series = pd.Series(recall, name='Recall')
    mcc_series = pd.Series(mcc, name='MCC')
    f1_series = pd.Series(f1, name='F1')
    cohen_kappa_series = pd.Series(cohen_kappa, name='Cohen Kappa')

    df = pd.concat([acc_series, balanced_series, precision_series,
                    recall_series, mcc_series, f1_series, cohen_kappa_series], axis=1)
    return df

# Calculate Confusion Matrix
def calc_confusion_matrix(input_data):
    Y_actual = input_data.iloc[:,0]
    Y_predicted = input_data.iloc[:,1]
    confusion_matrix_array = confusion_matrix(Y_actual, Y_predicted)
    confusion_matrix_df = pd.DataFrame(confusion_matrix_array, columns=['Actual', 'Predicted'], index=['Actual', 'Predicted'])
    return confusion_matrix_df

# Load example data
def load_example_data():
    df = pd.read_csv('Y_example.csv')
    return df

# Download performance matrics
def file_download(df):
    csv = df.to_csv(index=False)
    # sttring <-> bytes conversion
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href = "data:file/csv;base64,{b64}" download="performance_matrics.csv">Download CSV</a>'
    return href


# Sidebar -- Header
st.sidebar.header('Input Panel')
st.sidebar.markdown("""
[Example CSV file](https://raw.githubusercontent.com/dataprofessor/model_performance_app/main/Y_example.csv)
""")

# Sidebar pannel - Upload Input file
uploaded_file = st.sidebar.file_uploader(
    'Upload your input csv file', type=['csv'])

# Sidebar pannel performance matrics
performance_matrics = ['Accuracy', 'Balanced Accuracy',
                       'Precision', 'Recall', 'MCC', 'F1', 'Cohen Kappa']
selected_matrics = st.sidebar.multiselect('Performance Matrics', performance_matrics, performance_matrics)

# Main Panel
image = Image.open('logo.png')
st.image(image, width = 500)
st.title('Model Performance Calculator App')
st.markdown("""
This app calculates the model performance matrics for given actual and predicted values.
* **Python libraries:** `base64`, `pandas`, `streamlit`, `scikit-learn`
""")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    confusion_matrix_df = calc_confusion_matrix(input_df)
    matrix_df = calc_metrics(input_df)
    selected_matrics_df = matrix_df[ selected_matrics ]
    st.header('Input data')
    st.write(input_df)
    st.header('Confusion Matrix')
    st.write(confusion_matrix_df)
    st.header('Performance Matrics')
    st.write(selected_matrics_df)
    st.markdown(file_download(selected_matrics_df), unsafe_allow_html = True)
else:
    st.info('Awating the upload of the input file.')
    if st.button('Use Example Data'):
        input_df = load_example_data()
        confusion_matrix_df = calc_confusion_matrix(input_df)
        matrix_df = calc_metrics(input_df)
        selected_matrics_df = matrix_df[ selected_matrics ]
        st.header('Input data')
        st.write(input_df)
        st.header('Confusion Matrix')
        st.write(confusion_matrix_df)
        st.header('Performance Matrics')
        st.write(selected_matrics_df)
        st.markdown(file_download(selected_matrics_df), unsafe_allow_html = True)
