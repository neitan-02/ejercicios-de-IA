from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from audio import procesar_audio
from correo import enviar_correo

app = Flask(__name__)

# Carpeta para guardar archivos de audio temporalmente
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Función para dar formato bonito a la respuesta JSON
def formatear_respuesta_bonita(respuesta):
    enviados = respuesta.get("enviados", [])
    errores = respuesta.get("errores", [])

    mensaje = "📧 Resultado del envío de correos:\n"
    if enviados:
        for envio in enviados:
            partes = envio.split(" a ")
            correo = partes[-1] if len(partes) > 1 else envio
            mensaje += f"- ✅ {correo}\n"
    else:
        mensaje += "- ❌ No se enviaron correos.\n"

    if errores:
        mensaje += "\n⚠️ Errores:\n"
        for error in errores:
            mensaje += f"- {error}\n"

    return mensaje

# Ruta para recibir el archivo de audio
@app.route("/audio", methods=["POST"])
def audio():
    if "audio" not in request.files:
        return jsonify({"error": "No se envió un archivo de audio"}), 400

    archivo = request.files["audio"]
    if archivo.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    if not (archivo.filename.endswith(".mp3") or archivo.filename.endswith(".m4a")):
        return jsonify({"error": "Formato no soportado. Usa .mp3 o .m4a"}), 400

    filename = secure_filename(archivo.filename)
    ruta_audio = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        # Guardar archivo temporalmente
        archivo.save(ruta_audio)
        print(f"✅ Audio guardado en: {ruta_audio}")

        # Procesar el archivo de audio
        resultados = procesar_audio(ruta_audio)
        if not isinstance(resultados, list):
            resultados = [resultados]

        enviados, errores = [], []

        for item in resultados:
            destinatario = item.get("destinatario")
            asunto = item.get("asunto")
            mensaje = item.get("mensaje")

            if destinatario and asunto and mensaje:
                if enviar_correo(destinatario, asunto, mensaje):
                    enviados.append(f"Correo enviado a {destinatario}")
                else:
                    errores.append(f"Error al enviar correo a {destinatario}")
            else:
                errores.append(f"Faltan datos para {destinatario or 'destinatario desconocido'}")

        respuesta = {"enviados": enviados, "errores": errores}
        mensaje_formateado = formatear_respuesta_bonita(respuesta)

        return jsonify({
            "mensaje": mensaje_formateado,
            "json": respuesta
        }), 200 if not errores else 207

    except Exception as e:
        print(f"❌ Error al procesar la solicitud: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Eliminar archivo temporal
        if os.path.exists(ruta_audio):
            os.remove(ruta_audio)
            print(f"🧹 Archivo eliminado: {ruta_audio}")

# Ejecutar servidor Flask
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
