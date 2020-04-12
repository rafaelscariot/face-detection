import cv2

cascade_path = '../../resources/haarcascade_frontalface_default.xml'

def main():
    clf = cv2.CascadeClassifier(cascade_path)

    capture = cv2.VideoCapture(2)

    while True:
        _, frame = capture.read()

        faces = clf.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Detected face.', (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main().run()