import cv2
import dlib

detector = dlib.get_frontal_face_detector()


def main():
    capture = cv2.VideoCapture(2)
    
    while True:
        _, frame = capture.read()

        faces = detector(frame)
        
        for face in faces:
            l, t, r, b = (
                face.left(),
                face.top(),
                face.right(),
                face.bottom()
            )

            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, 'Detected face.', (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    main().run()
    
