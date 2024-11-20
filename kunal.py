from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import pandas as pd
import tempfile
import sweetviz as sv
import ydata_profiling as pp
import chardet
from app.insights import InsightsGenerator

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"

# Initialize Insights Generator
insights_generator = InsightsGenerator()

# Helper function to detect encoding
def detect_encoding(file):
    raw_data = file.read()
    file.seek(0)
    detected_encoding = chardet.detect(raw_data)["encoding"]
    return detected_encoding

# Function to generate ydata-profiling report
def generate_ydata_profiling_report(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    profile = pp.ProfileReport(df, title="YData Profiling Report", explorative=True)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    profile.to_file(temp_file.name)
    return temp_file.name

# Function to generate Sweetviz report
def generate_sweetviz_report(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    report = sv.analyze(df)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    report.show_html(temp_file.name)
    return temp_file.name

# Route for home page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files.get("dataset")
        if uploaded_file and uploaded_file.filename.endswith(".csv"):
            try:
                encoding = detect_encoding(uploaded_file)
                df = pd.read_csv(uploaded_file, encoding=encoding)
                df.to_csv("uploaded_data.csv", index=False)  # Save uploaded file temporarily
                flash("Dataset uploaded successfully!", "success")
                return redirect(url_for("eda", dataset_name=uploaded_file.filename))
            except Exception as e:
                flash(f"Error processing file: {e}", "danger")
        else:
            flash("Please upload a valid CSV file.", "danger")
    return render_template("index.html")

# Route for EDA
@app.route("/eda/<dataset_name>", methods=["GET", "POST"])
def eda(dataset_name):
    try:
        df = pd.read_csv("uploaded_data.csv")
        df_sample = df.sample(frac=0.1, random_state=42) if len(df) > 100000 else df

        # Generate statistics for the dataset
        data_info = {
            "Shape": df.shape,
            "Columns": df.columns.tolist(),
            "Data Types": df.dtypes.to_dict(),
            "Missing Values": df.isnull().sum().to_dict(),
            "Summary Stats": df.describe().to_dict(),
        }

        if request.method == "POST":
            eda_choice = request.form.get("eda_choice")
            if eda_choice == "ydata":
                report_path = generate_ydata_profiling_report(df_sample)
            else:
                report_path = generate_sweetviz_report(df_sample)
            return send_file(report_path, as_attachment=True)

        # Render the page with dataset sample and statistics
        return render_template(
            "report.html", 
            dataset_name=dataset_name, 
            df_sample=df_sample.head().to_html(classes="table"),
            data_info=data_info  # Pass statistics to template
        )
    except Exception as e:
        flash(f"Error generating EDA: {e}", "danger")
        return redirect(url_for("index"))

# Route for insights page
@app.route("/insights/<dataset_name>", methods=["GET", "POST"])
def insights(dataset_name):
    try:
        # Read the uploaded dataset
        df = pd.read_csv("uploaded_data.csv")

        # Use the InsightsGenerator to generate insights for the dataset
        insights_report = insights_generator.generate(df)

        # Return insights (you can display them directly or generate a downloadable report)
        return render_template("insights.html", insights=insights_report)

    except Exception as e:
        flash(f"Error generating insights: {e}", "danger")
        return redirect(url_for("index"))

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
