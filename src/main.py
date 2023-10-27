# main.py
from src.vision import Vision
from src.speech import Speech


def main():
    vision_model_path = 'models/yolov8n.pt'
    voice_model_name = 'facebook/mms-tts-por'
    image_path = 'resources/test_cena1.jpg'

    vision = Vision(vision_model_path)
    speech = Speech(voice_model_name)

    detection_result = vision.detect_objects(image_path)

    message = analyze_detection(detection_result)

    audio_output_path = 'outputs/output_audio.wav'
    speech.text_to_speech(message, audio_output_path)


def analyze_detection(detection_result):
    # Implemente a lógica para determinar a mensagem com base nos resultados de detecção.
    # Por exemplo, se um semáforo foi detectado:
    return 'Semaforo detectado. Cuidado!'


if __name__ == '__main__':
    main()
