import torch
import librosa
import os
import re
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
from pydub.utils import which
from correo import enviar_correo

AudioSegment.converter = which("ffmpeg")
FFMPEG_PATH = "C:/ffmpeg/bin/ffmpeg.exe"  

model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

def convert_to_wav(m4a_path):
    wav_path = m4a_path.replace(".m4a", ".wav")
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format="wav")
    return wav_path

def load_audio(file):
    try:
        if file.endswith(".m4a"):
            print("Convirtiendo archivo .m4a a .wav...")
            file = convert_to_wav(file)
        print(f"Cargando: {file}")
        waveform, sr = librosa.load(file, sr=16000)
        if file.endswith(".wav") and os.path.exists(file):
            os.remove(file)
        return torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        print(f"Error cargando audio: {e}")
        raise

def corregir_errores(texto):
    texto = texto.lower()
    correcciones = {
    
        "alaura": "a laura",
        "aenrique": "a enrique",
        "asofia": "a sofia",
        "ala portda": "haz la portada",
        "ala portada": "haz la portada",

        "correr": "correo",
        "correoo": "correo",
        "corre": "correo",
        "coreo": "correo",

        "guasap": "whatsapp",
        "güasap": "whatsapp",
        "wasap": "whatsapp",

        "imeil": "email",
        "güimail": "gmail",
        "envía": "enviar",
        "envié": "enviar",
        "enviar un wasap": "enviar whatsapp",
        "enviar un imeil": "enviar correo",
    }
    for error, correccion in correcciones.items():
        texto = texto.replace(error, correccion)
    return texto

def normalizar_destinatario(nombre):
    correcciones = {
        "laora": "laura",
        "enrrique": "enrique",
        "sofia": "sofia",
        "sofía": "sofia",
    }
    nombre = nombre.lower().strip()
    return correcciones.get(nombre, nombre)

def audio_to_text(file_path):
    input_audio = load_audio(file_path)
    inputs = processor(input_audio.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    text = transcription[0].lower()
    texto_corregido = corregir_errores(text)
    return texto_corregido

#Procesar múltiples destinatarios e instrucciones
def procesar_comando_correo_multiple(texto):
    instrucciones = re.findall(r"a (\w+): ([^\.]+)\.", texto)
    asunto_match = re.search(r"asunto: ([^\.]+)", texto)

    if not instrucciones:
        return {"destinatario": None, "asunto": None, "mensaje": "❌ No se detectaron instrucciones válidas."}

    asunto = asunto_match.group(1).strip() if asunto_match else "Sin asunto"
    resultados = []

    for nombre, tarea in instrucciones:
        nombre = normalizar_destinatario(nombre)
        mensaje = tarea.strip().capitalize()
        exito = enviar_correo(nombre, asunto, mensaje)

        resultados.append({
            "destinatario": nombre,
            "mensaje": mensaje,
            "enviado": exito
        })

    return {
        "destinatario": [r["destinatario"] for r in resultados],
        "asunto": asunto,
        "mensaje": resultados
    }


def procesar_comando_correo_simple(texto):
    try:
        if " a " not in texto:
            return {"destinatario": None, "asunto": None, "mensaje": "❌ No se detectó destinatario válido."}

        parte_despues_de_a = texto.split(" a ", 1)[1]
        nombre_destinatario = parte_despues_de_a.split(" ", 1)[0].strip().lower()
        nombre_destinatario = normalizar_destinatario(nombre_destinatario)

        if "que le diga" in texto:
            cuerpo_mensaje = texto.split("que le diga", 1)[1].strip().capitalize()
        else:
            cuerpo_mensaje = "Este es un mensaje enviado por voz."

        asunto = "Mensaje por voz"
        resultado_envio = enviar_correo(nombre_destinatario, asunto, cuerpo_mensaje)

        if resultado_envio:
            return {
                "destinatario": nombre_destinatario,
                "asunto": asunto,
                "mensaje": f"✅ Correo enviado a {nombre_destinatario}: {cuerpo_mensaje}"
            }
        else:
            return {
                "destinatario": nombre_destinatario,
                "asunto": asunto,
                "mensaje": f"❌ Error al enviar correo a {nombre_destinatario}"
            }

    except Exception as e:
        return {"destinatario": None, "asunto": None, "mensaje": f"❌ Error procesando comando simple: {str(e)}"}

# ACTUALIZADO: detectar múltiples o simples
def detectar_comando(texto):
    texto = texto.lower()
    print(f"[detectar_comando] Texto recibido: {texto}")

    if texto.count("a ") > 1 and "asunto:" in texto:
        return procesar_comando_correo_multiple(texto)

    return procesar_comando_correo_simple(texto)

def procesar_audio(ruta_audio):
    
    texto = "a laura revisa la parte financiera a enrique modifica el cronograma a sofia haz la portada Asunto: entrega de proyecto"
    texto = texto.lower()

    
    destinatarios_email = {
        "laura": "laura@gmail.com",
        "enrique": "enrique@gmail.com",
        "sofia": "sofia@gmail.com"
    }

    resultados = []
    for nombre, correo in destinatarios_email.items():
        
        if f"a {nombre}" in texto:
          
            partes = texto.split(f"a {nombre}")
            if len(partes) > 1:
                fragmento = partes[1]
                
                for otro_nombre in destinatarios_email:
                    if otro_nombre != nombre:
                        fragmento = fragmento.split(f"a {otro_nombre}")[0]
                mensaje = fragmento.strip()
                if mensaje:
                    resultados.append({
                        "destinatario": correo,
                        "asunto": "Mensaje por voz",
                        "mensaje": f"{mensaje}"
                    })

    if not resultados:
        
        return [{
            "destinatario": None,
            "asunto": None,
            "mensaje": "❌ No se detectó destinatario válido."
        }]

    return resultados

if __name__ == "__main__":
    ruta = "./prueba_audio.m4a"
    print(procesar_audio(ruta))