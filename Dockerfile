# Usamos una imagen de Python compatible con TensorFlow y ligera
FROM python:3.10-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los requisitos e instalar dependencias
# Usamos --no-cache-dir para ahorrar espacio de disco
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto (Hugging Face espera que la app se ejecute en 7860)
EXPOSE 7860

# Comando para iniciar la app optimizado para memoria
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "1", "--threads", "1", "app:app"]