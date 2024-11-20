import os
import ydata_profiling as pp
import sweetviz as sv

class EDAReport:
    @staticmethod
    def generate_ydata_profiling_report(df, output_file="templates/eda_report.html"):
        # Check if the directory exists, if not, create it
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            # Generate the ydata profiling report
            profile = pp.ProfileReport(df, title="YData Profiling Report", explorative=True)
            # Save the report to the specified file
            profile.to_file(output_file)
            return output_file
        except Exception as e:
            print(f"Error generating the EDA report: {e}")
            return None

    @staticmethod
    def generate_sweetviz_report(df, output_file="templates/sweetviz_report.html"):
        try:
            # Generate Sweetviz report
            report = sv.analyze(df)
            report.show_html(output_file)
            return output_file
        except Exception as e:
            print(f"Error generating the Sweetviz report: {e}")
            return None
