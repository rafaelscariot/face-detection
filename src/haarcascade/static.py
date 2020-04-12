import cv2

image_path = '../../images/person.jpeg'
cascade_path = '../../resources/haarcascade_frontalface_default.xml'

def main():
    clf = cv2.CascadeClassifier(cascade_path)
    img = cv2.imread(image_path)
    faces = clf.detectMultiScale(img, 1.3, 10)

    for (x, y, h, w) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, 'Detected face.', (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main().run()