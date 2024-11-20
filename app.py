import streamlit as st
import pandas as pd
import tempfile
import sweetviz as sv
import ydata_profiling as pp
from app.insights import InsightsGenerator
import os
import chardet

# Initialize Insights Generator
insights_generator = InsightsGenerator()

# App title
st.set_page_config(page_title="Automated EDA App", layout="wide")
st.title("📊 Automated EDA with Kunal")

# Function to generate ydata-profiling report
def generate_ydata_profiling_report(df):
    try:
        # Convert object columns to category type to avoid Arrow issues
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')

        # Generate the ydata profiling report
        profile = pp.ProfileReport(df, title="YData Profiling Report", explorative=True)
        
        # Save the report to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        profile.to_file(temp_file.name)
        return temp_file.name  # Return the path to the generated report
    except Exception as e:
        st.error(f"Error generating YData Profiling report: {e}")
        return None

# Function to generate Sweetviz report
def generate_sweetviz_report(df):
    try:
        # Convert object columns to category type to avoid Arrow issues
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')

        # Generate the Sweetviz report
        report = sv.analyze(df)
        
        # Save the Sweetviz report to a temporary file
        report_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        report.show_html(report_file.name)
        return report_file.name
    except Exception as e:
        st.error(f"Error generating Sweetviz report: {e}")
        return None

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    # Handle file encoding errors
    try:
        # Detect the encoding of the file using chardet
        raw_data = uploaded_file.read()
        detected_encoding = chardet.detect(raw_data)['encoding']
        
        # Reset file pointer to the beginning of the file
        uploaded_file.seek(0)
        
        # Load the dataset using the detected encoding
        df = pd.read_csv(uploaded_file, encoding=detected_encoding)

        # Handle the Arrow serialization issue: convert columns to appropriate types
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')

    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        df = pd.DataFrame()  # Set df as empty in case of error

    # Check if the DataFrame is empty
    if df.empty:
        st.error("The dataset is empty. Please upload a valid CSV file.")
    else:
        # Display a warning if the dataset is too large
        if df.shape[0] > 100000:  # Adjust the threshold based on memory constraints
            st.warning("The dataset is large. Only a sample will be used for EDA.")

            # Sample the dataset for large files
            df_sample = df.sample(frac=0.1, random_state=42)  # Taking 10% sample, adjust as needed
        else:
            df_sample = df

        # Show dataset preview
        st.write("### Dataset Preview")
        st.write(df_sample.head())

        # EDA Options
        st.write("### Generate EDA Report")
        eda_choice = st.radio("Choose EDA Method", ("YData Profiling", "Sweetviz"))
        if st.button("Generate EDA Report"):
            try:
                if eda_choice == "YData Profiling":
                    report_path = generate_ydata_profiling_report(df_sample)
                else:
                    report_path = generate_sweetviz_report(df_sample)
                
                if report_path:
                    # Embed the HTML report into the app
                    with open(report_path, "r", encoding="utf-8") as report_file:
                        report_html = report_file.read()
                        st.components.v1.html(report_html, height=800)
                    
                    # Provide download button for the report
                    with open(report_path, "rb") as report_file:
                        st.download_button(
                            label="Download Report",
                            data=report_file,
                            file_name="eda_report.html",
                            mime="text/html"
                        )
                    
                    # Optionally, delete the temporary report file after displaying it
                    os.remove(report_path)

            except Exception as e:
                st.error(f"Error generating the report: {e}")

        # AI-Generated Insights
        st.write("### AI-Generated Insights")
        input_text = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns. Summarize key patterns."
        try:
            insights = insights_generator.generate_insights(input_text)
            st.write(insights)
        except Exception as e:
            st.error(f"Error generating AI insights: {e}")

        # Data Summary
        st.write("### Summary Statistics")
        st.write(df_sample.describe())
