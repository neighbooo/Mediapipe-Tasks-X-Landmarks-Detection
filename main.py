import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

from enum import Enum
from visualization import draw_landmarks_on_image

camera = cv2.VideoCapture(0)
winname = "Window"

class Parameters(Enum):
    base_options = 'model.task'
    running_mode = 'LIVE_STREAM'
    num_hands = 2
    min_hand_detection_confidence = .7
    min_hand_presence_confidence = .7
    min_tracking_confidence = .7

def callback(result, output_image: mp.Image, timestamp_ms: int):
    annotated_image = draw_landmarks_on_image(output_image.numpy_view() ,result)
    global frame
    frame = annotated_image

base_options = mp.tasks.BaseOptions
options = vision.HandLandmarkerOptions(
    base_options = base_options(model_asset_path=Parameters.base_options.value),
    running_mode = vision.RunningMode(Parameters.running_mode.value),
    num_hands = Parameters.num_hands.value,
    min_hand_detection_confidence = Parameters.min_hand_detection_confidence.value,
    min_hand_presence_confidence = Parameters.min_hand_presence_confidence.value,
    min_tracking_confidence = Parameters.min_tracking_confidence.value,
    result_callback=callback
)

HandLandmarker = mp.tasks.vision.HandLandmarker
while(True):
    success, feed = camera.read()
    rgb_image = cv2.cvtColor(feed, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    timestamp = int(camera.get(cv2.CAP_PROP_POS_MSEC)*1000)

    with HandLandmarker.create_from_options(options) as detector:
        parsed_image = detector.detect_async(mp_image, timestamp)
    #--

    cv2.imshow(winname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    #-Detect Escape-
    key = cv2.waitKey(1)
    if key == ord(chr(27)):
        break

camera.release()
cv2.destroyAllWindows()