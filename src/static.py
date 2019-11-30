import cv2

image_path = '../images/pic.jpeg'
cascade_path = '../haarcascade_frontalface_default.xml'

def main():
    clf = cv2.CascadeClassifier(cascade_path)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray, 1.3, 10)

    for (x, y, h, w) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main().run()