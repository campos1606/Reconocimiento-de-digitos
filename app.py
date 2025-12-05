import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS 
from tensorflow.keras.models import load_model
from supabase import create_client, Client
from dotenv import load_dotenv
from postgrest.exceptions import APIError

# Cargar variables de entorno
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Inicializar Supabase
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Error inicializando Supabase: {e}")
    
# Inicializar Flask y el modelo de ML
app = Flask(__name__)
CORS(app) 
MODEL = load_model('digit_recognizer_model.h5')

def preprocess_image(image_data_url):
    """
    Decodifica la imagen base64, la procesa, la redimensiona a 28x28 y la normaliza.
    """
    encoded_data = image_data_url.split(',')[1]
    image_bytes = base64.b64decode(encoded_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # 1. Convertir a escala de grises
    image = image.convert('L')
    
    # 2. Redimensionar a 28x28
    image = image.resize((28, 28))
    
    # 3. Convertir a array de NumPy y normalizar (0.0 a 1.0)
    img_array = np.array(image).astype('float32') / 255.0
    
    # 4. Reformar para Keras
    processed_img = np.expand_dims(img_array, axis=0)
    processed_img = np.expand_dims(processed_img, axis=-1)
    
    return processed_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data_url = data['image_data']
        
        processed_img = preprocess_image(image_data_url)
        
        predictions = MODEL.predict(processed_img, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class] * 100

        python_confidence = float(round(confidence, 2))

        data_to_insert = {
            "digit_recognized": int(predicted_class),
            "confidence_score": python_confidence,
            "image_base64": image_data_url
        }
        supabase.table("digits").insert(data_to_insert).execute()

        return jsonify({
            'success': True,
            'predicted_digit': int(predicted_class),
            'confidence': python_confidence
        })
        
    except Exception as e:
        print(f"Error en la predicción o Supabase: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    try:
        response = supabase.table("digits").select("*").order("created_at", desc=True).limit(10).execute()
        
        if hasattr(response, 'data') and response.data is not None:
            history_data = response.data
            return jsonify({
                'success': True,
                'history': history_data
            })
        else:
            return jsonify({'success': True, 'history': []})

    except Exception as e:
        print(f"Error al obtener historial de Supabase: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/delete_history', methods=['POST'])
def delete_history():
    try:
        # Usamos .delete().neq() en lugar de truncate
        supabase.table("digits").delete().neq("digit_recognized", -1).execute() 
        return jsonify({'success': True, 'message': 'Historial eliminado correctamente.'})
    except APIError as e:
        print(f"Error de Supabase al eliminar: {e}")
        return jsonify({'success': False, 'error': f"Error de la base de datos: {e.message}"}), 500
    except Exception as e:
        print(f"Error general al eliminar historial: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)