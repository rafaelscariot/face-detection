import cv2
import dlib


def main():
    image_path = '../../images/person.jpeg'
    detector = dlib.get_frontal_face_detector()

    img = cv2.imread(image_path)
    faces = detector(img)

    for face in faces:
        l, t, r, b = (
            face.left(),
            face.top(),
            face.right(),
            face.bottom()
        )

        cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(img, 'Detected face.', (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    main().run()
    
