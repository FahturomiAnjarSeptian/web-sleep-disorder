from flask import Flask, render_template, request
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg') # PENTING: Agar matplotlib jalan di server tanpa layar
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# 1. LOAD MODEL
try:
    with open('model_sleep.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    forest = saved_data['forest']
    X_min = saved_data['X_min']
    X_max = saved_data['X_max']
    feature_names = saved_data['feature_names']
except FileNotFoundError:
    print("Model belum dilatih! Jalankan train_model.py dulu.")

# 2. FUNGSI PREDIKSI MANUAL (Wajib ada di sini untuk membaca struktur tree)
def predict_tree(tree, x):
    if 'label' in tree: return tree['label']
    if x[tree['feature']] < tree['threshold']:
        return predict_tree(tree['left'], x)
    else:
        return predict_tree(tree['right'], x)

def predict_forest(trees, X_input):
    # X_input shape (1, n_features)
    preds = [predict_tree(t, X_input[0]) for t in trees]
    return int(np.round(np.mean(preds)))

# 3. FUNGSI GAMBAR POHON (Dimodifikasi untuk Web/Base64)
def get_tree_image(tree, feature_names, title):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_title(title, fontsize=12)
    ax.axis("off")

    def recurse(node, x=0.5, y=1.0, dx=0.25, dy=0.15):
        if 'label' in node:
            ax.text(x, y, f"Leaf\nPred={int(node['label'])}", 
                    ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="#90EE90"))
            return

        feat = node['feature']
        thr = node['threshold']
        feat_name = feature_names[feat]

        ax.text(x, y, f"{feat_name}\n< {thr:.2f}", 
                ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", fc="#ADD8E6"))

        # Garis ke kiri (True)
        ax.plot([x, x - dx], [y - 0.02, y - dy + 0.02], color="black")
        recurse(node['left'], x - dx, y - dy, dx * 0.5, dy)

        # Garis ke kanan (False)
        ax.plot([x, x + dx], [y - 0.02, y - dy + 0.02], color="black")
        recurse(node['right'], x + dx, y - dy, dx * 0.5, dy)

    recurse(tree)
    
    # Simpan ke buffer memory (bukan plt.show)
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close() # Bersihkan memori
    return plot_url

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    tree_plots = [] # List untuk menyimpan semua gambar pohon (Opsi B)
    
    if request.method == 'POST':
        try:
            # Ambil data dari form HTML
            # Urutan harus SAMA PERSIS dengan numeric_cols di train_model.py
            # ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 
            # 'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate', 
            # 'Daily Steps', 'Systolic BP', 'Diastolic BP']
            
            # Mapping manual input dropdown ke angka
            gender = int(request.form['gender'])
            age = float(request.form['age'])
            occup = int(request.form['occupation'])
            sleep_dur = float(request.form['sleep_duration'])
            qual_sleep = float(request.form['quality_sleep'])
            phys_act = float(request.form['phys_activity'])
            stress = float(request.form['stress'])
            bmi = int(request.form['bmi'])
            heart_rate = float(request.form['heart_rate'])
            steps = float(request.form['daily_steps'])
            systolic = float(request.form['systolic'])
            diastolic = float(request.form['diastolic'])

            # Buat array input
            input_data = np.array([[gender, age, occup, sleep_dur, qual_sleep, 
                                    phys_act, stress, bmi, heart_rate, steps, 
                                    systolic, diastolic]])
            
            # Normalisasi Input (PENTING!)
            input_scaled = (input_data - X_min) / (X_max - X_min + 1e-8)

            # Prediksi
            prediction = predict_forest(forest, input_scaled)
            
            result_label = "Ada Gangguan Tidur (Sleep Apnea/Insomnia)" if prediction == 1 else "Normal / Tidak Ada Gangguan"
            color_class = "text-danger" if prediction == 1 else "text-success"
            prediction_text = f"Hasil Prediksi: <span class='{color_class}'><strong>{result_label}</strong></span>"

            # VISUALISASI SEMUA POHON (OPSI B)
            for i, tree in enumerate(forest):
                title = f"Visualisasi Pohon Keputusan ke-{i+1}"
                img_data = get_tree_image(tree, feature_names, title)
                tree_plots.append(img_data)

        except Exception as e:
            prediction_text = f"Error: {str(e)}. Pastikan semua field terisi angka."

    return render_template('index.html', prediction_text=prediction_text, tree_plots=tree_plots)