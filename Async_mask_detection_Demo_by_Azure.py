import os
import time
import multiprocessing as mp
import asyncio
import cv2
import azure.cognitiveservices.speech as speechsdk

from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials


# Global Settings
PIC_FOLDER = './record/'
DETECT_CYCLE = 3    # 1 face detect call per 3 sec
FPS = 40            # 40 frame per second
dltT = 1/FPS        # deltaT: 0.025s per frame


def analysis(file_path):
    SKEY = '7bd508c7be31425fc34d8d4b....' # 填寫自己的key
    ENDP = 'https://faceronson.cognitiveservices.azure.com/'

    # Create an authenticated FaceClient.
    face_client = FaceClient(ENDP, CognitiveServicesCredentials(SKEY))

    # Image of face(s)
    face_attributes = ['Occlusion'] # Occulusion 是紀錄臉部障礙物是否有的參數
    img = open(file_path, "r+b")

    # Detect a face with attributes, returns a list[DetectedFace]
    detected_faces = face_client.face.detect_with_stream(
                           image = img,
          return_face_attributes = face_attributes,
               recognition_model = 'recognition_03',
        return_recognition_model = False,
                 detection_model = 'detection_01',
                  custom_headers = None,
                             raw = False,
                        callback = None)
    if detected_faces:
        for face in detected_faces:
            if not face.face_attributes.occlusion.mouth_occluded:  # mouth_occluded 針對嘴部的障礙物去做偵測
                return "Please wear face mask right now !"
    return None


def text_to_audio(sentence):
    text2audio = sentence
    speech_key, service_region = "94f99627b5f54e41b9987f0c8.....", "southcentralus"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    # Creates a speech synthesizer using the default speaker as audio output.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config = speech_config)

    # Synthesizes the received text to speech.
    # The synthesized speech is expected to be heard on the speaker with this line executed.
    result = speech_synthesizer.speak_text_async(text2audio).get()

    # Checks result.
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized to speaker for text [{}]".format(text2audio))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
        print("Did you update the subscription info?")


def detectAP(q):
    while True:
        try:
            fName = q.get()
            if fName == 'q':
                break
            print('E:{}'.format( time.time() ), end=' ')
            retext = analysis(fName)
            if retext != None:
                # print(retext, ' ')
                text_to_audio(retext)
            print('X:{}'.format( time.time() ))
        except Exception as error:
            print('sleep 0.05')
            time.sleep(0.05)
    print('Exit detect Loop')


async def main():
    ctx = mp.get_context('spawn')
    que = ctx.Queue()
    ap1 = ctx.Process(target=detectAP, args=(que,))
    ap1.start()
    cap = cv2.VideoCapture(0)
    tmLastSave = time.time()

    while(True):
        tmLpStart = time.time()
        ret, save_img = cap.read()

        if ret == True:
            cv2.imshow("image", save_img)
            if DETECT_CYCLE - (time.time() - tmLastSave) <= 0:
                tmLastSave = time.time()
                print(f'SAVE: {tmLastSave}', end=' ')
                fName = PIC_FOLDER + 'check.jpg'
                if cv2.imwrite(fName, save_img):
                    que.put(fName)

        key = cv2.waitKey(1)    # 延遲一毫秒，並取得按鍵
        if key & 0xFF == ord('q'):
            print("disable camera")
            while not que.empty():
                que.get()
            que.put('q')
            break
        dts = dltT - (time.time() - tmLpStart)
        if dts > 0:
            await asyncio.sleep(dts)

    cap.release()
    cv2.destroyAllWindows()
    await asyncio.sleep(0.1)
    que.close()
    ap1.join()


if __name__ == "__main__" :
    if not os.path.exists(PIC_FOLDER):
        os.makedirs(PIC_FOLDER, exist_ok=True)
    asyncio.run(main())

