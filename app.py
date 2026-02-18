from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_required, login_user
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression
import numpy as np
from transformers import pipeline
from reportlab.pdfgen import canvas
from flask import send_file

app = Flask(__name__)
app.secret_key = 'your-secret-key-2026-nathiya'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

def generate_bar_chart(x, y, title, xlabel, ylabel, color='skyblue'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, y, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(y):
        ax.text(i, v + 0.5, f"{v:.1f}", ha='center')
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    return plot_url

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        if username:  # Simple login - any username works
            user = User(username)
            login_user(user)
            return redirect('/')
        flash('Invalid login', 'danger')
    return render_template('login.html')

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename.endswith('.csv'):
            return render_template('upload.html', error="Please upload a valid CSV file")

        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render_template('upload.html', error=f"Error reading CSV: {str(e)}")

        df.columns = df.columns.str.strip()
        non_subject_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        subject_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

        if not subject_cols:
            return render_template('upload.html', error="No numeric subject columns found in CSV")

        stats = df[subject_cols].agg(['mean', 'max', 'min']).T
        stats.columns = ['Average', 'Highest', 'Lowest']
        stats = stats.round(2)

        df['Average'] = df[subject_cols].mean(axis=1).round(2)
        ranked = df.sort_values('Average', ascending=False).reset_index(drop=True)
        ranked['Rank'] = ranked.index + 1
        ranked = ranked[non_subject_cols + ['Average', 'Rank'] + subject_cols]

        threshold = 70
        at_risk = ranked[ranked['Average'] < threshold][non_subject_cols + ['Average']]

        # AI Prediction
        if len(subject_cols) >= 2:
            X = df[subject_cols[:-1]].values
            y = df[subject_cols[-1]].values
            model = LinearRegression().fit(X, y)
            
            # Dynamically create example input with correct number of features
            num_features = X.shape[1]  # number of columns the model expects
            example_input = np.array([[75] * num_features])  # use 75 as neutral example score
            
            predicted = model.predict(example_input)[0]
            predicted_msg = f"AI Predicted score for new student (example input): {predicted:.2f}"
        else:
            predicted_msg = "Not enough subjects for prediction"
        # Charts
        subject_plot = generate_bar_chart(stats.index, stats['Average'], "Average Scores per Subject", "Subjects", "Score")
        top_n = min(10, len(ranked))
        student_plot = generate_bar_chart(ranked.head(top_n).get('Name', ranked.head(top_n).index.astype(str)), ranked.head(top_n)['Average'], "Top Students", "Student", "Average")

        # Save for PDF export
        session['df'] = df.to_json()
        session['stats'] = stats.to_json()
        session['ranked'] = ranked.to_json()
        session['at_risk'] = at_risk.to_json()

        tips = {
            "Low Average Students": f"There are {len(at_risk)} students below {threshold}%. Consider extra classes.",
            "Weakest Subject": f"The lowest average is in {stats['Average'].idxmin()} ({stats['Average'].min():.1f}). Focus here.",
            "General Advice": "Encourage daily revision and practice tests.",
            "Next Steps": "Upload attendance or behavior data for better predictions."
        }

        return render_template('result.html', tables={'stats': stats.to_html(classes='table table-striped', index=True), 'ranked': ranked.to_html(classes='table table-striped', index=False), 'at_risk': at_risk.to_html(classes='table table-striped', index=False) if not at_risk.empty else None}, threshold=threshold, subject_plot=subject_plot, student_plot=student_plot, student_count=len(df), subject_count=len(subject_cols), predicted_msg=predicted_msg, tips=tips)

    return render_template('upload.html')

@app.route('/export_pdf')
@login_required
def export_pdf():
    df = pd.read_json(session.get('df', '{}'))
    stats = pd.read_json(session.get('stats', '{}'))
    ranked = pd.read_json(session.get('ranked', '{}'))
    at_risk = pd.read_json(session.get('at_risk', '{}'))

    output = io.BytesIO()
    p = canvas.Canvas(output)
    p.drawString(100, 800, "Student Performance Report")
    p.drawString(100, 780, "Stats: " + str(stats))
    p.drawString(100, 760, "Ranked: " + str(ranked))
    p.drawString(100, 740, "At Risk: " + str(at_risk))
    p.save()
    output.seek(0)
    return send_file(output, as_attachment=True, download_name='report.pdf')

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)