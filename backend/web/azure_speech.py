import locale
import azure.cognitiveservices.speech as speechsdk

# Replace with your own subscription key and service region (e.g., "westus").
SUBSCRIPTION_KEY = "7683307099954c9d8b87c8e6d36e53c2"
SERVICE_REGION = "eastus"

# Set up the speech configuration

def getRating(file_name, lang):

    speech_config = speechsdk.SpeechConfig(subscription=SUBSCRIPTION_KEY, region=SERVICE_REGION)


    # Create a pronunciation assessment configuration
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text="",
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme)

    # Create a recognizer with the given settings
    audio_config = speechsdk.audio.AudioConfig(filename=file_name)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config, language=lang)

    # Apply the pronunciation assessment configuration to the recognizer
    pronunciation_config.apply_to(recognizer)

    print("Processing the WAV file...")

    # Start the recognition
    result = recognizer.recognize_once()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
        pronunciation_result = speechsdk.PronunciationAssessmentResult(result)
        print("Pronunciation Assessment Result:")
        print("Accuracy score: {}".format(pronunciation_result.accuracy_score))
        print("Fluency score: {}".format(pronunciation_result.fluency_score))
        print("Completeness score: {}".format(pronunciation_result.completeness_score))
    else:
        print("No speech could be recognized: {}".format(result.no_match_details))