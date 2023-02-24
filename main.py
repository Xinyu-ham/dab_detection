import cv2, argparse
from posturedetection import PoseDetector
from utils import get_random_insults

VID_FILE = 'assets/me_dabbin.mp4'
# VID_DIM = (360, 640)
VID_DIM = (720, 1280)

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--webcam', help="Whether to us webcam", action='store_true')
if parser.parse_args().webcam:
    vid_source = 0
else:
    vid_source = VID_FILE

# Instantiate model
pose_detector = PoseDetector(VID_DIM, mirror=True, angle_margin=25)
pose_detector.add_requirements('right_shoulder', 100)
pose_detector.add_requirements('left_shoulder', -110)
pose_detector.add_requirements('right_elbow', -5)
pose_detector.add_requirements('left_elbow', 179)



# Load video capture
cap = cv2.VideoCapture(vid_source)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w, h = int(cap.get(3)), int(cap.get(4))

out = cv2.VideoWriter('output/me_dabbin.mov', cv2.VideoWriter_fourcc(*'mp4v'), fps - 5, VID_DIM, True)

insult = get_random_insults()
dab_status = False
while cap.isOpened():
    # print('##################FRAME############\n')
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, VID_DIM)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # get landmarks
    pose_detector.detect(frame)
    is_dabbing = pose_detector.check_requirements()
    if is_dabbing:
        frame = cv2.putText(frame, insult, (50, 250), cv2.FONT_HERSHEY_PLAIN, 2, (10, 10, 10), 3)
  
    pose_detector.draw(frame)
    for part, (loc, angle, valid) in pose_detector.angles.items():
        color = (0, 255, 0) if valid else (255, 0, 0)
        frame = cv2.putText(frame, f'{angle}', loc, cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('vid', frame)
    out.write(frame)
    
    if dab_status and not is_dabbing:
        insult = get_random_insults()

    dab_status = is_dabbing
    if cv2.waitKey(1000 // fps) == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()