import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import pickle

def letters(img):
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    morph_letters = cv2.morphologyEx(img, cv2.MORPH_CROSS, rect_kernel) 
    cv2.imshow("morph", morph_letters)
    cnts = cv2.findContours(morph_letters.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    lengthOfContours = len(cnts)
    print("Number of counturs: ", lengthOfContours)
    points = []
    rects = []

    for (i, c) in enumerate(cnts):
        area = cv2.contourArea(c)
        if area > 80:
            print("area  "+str(area))
            (x, y, w, h) = cv2.boundingRect(c)
            x = x - 3
            y = y - 10
            w = w + 4
            h = h + 12
            cv2.rectangle(img_letters, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rect = [x, y, w+x, h+y]
            leftPoint = x
            #print("leftPoint: ", leftPoint)
            points.append(leftPoint)
            points.sort()
            rects.append([i,rect])
            rects.sort(key=lambda x: x[1])
            #print("rects: ", rects)
            if i == lengthOfContours-1:
            
                for i in range(len(rects)):
                    
                    found_letter = thresh_adaptive[rects[i][1][1]:rects[i][1][3], rects[i][1][0]:rects[i][1][2]]
                    prediction(found_letter)

def prediction(word_img):
    pick = open('model-letters.sav', 'rb')
    #pickle.dump(model, pick)
    model = pickle.load(pick)
    pick.close()

    #prediction = model.predict(x_test)
    #accuracy = model.score(x_test, y_test)

    categories = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J',  'K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','1','2','3','4',  '5','6','7','8','9','0']

    img = word_img
    try:
        img = cv2.resize(img, (50,50))
        img = np.array(img).flatten()
        img = img.reshape(1, -1)
        print("working")
        prediction = model.predict(img)
        letter = categories[prediction[0]]
        print("Prediction ", prediction)
        print("Prediction letter: ", letter)
        #print("Accuracy: ",accuracy)
        letterImg = img
        cv2.imshow("letter", letterImg)
        cv2.waitKey(0)

        fh = open("letters.txt", "a")
        fh.write(letter)
        fh.close()

    except:
        print("error")

index = -1
for i in range(30,-1,-1):
    index = index + 1
    dir = "images-lines2/line" + str(i) + ".png"
    img_orj = cv2.imread(dir)
    img_orj = cv2.resize(img_orj, None, fx=1, fy=1.5, interpolation=cv2.INTER_AREA)
    img_letters = img_orj.copy()
    gray = cv2.cvtColor(img_orj, cv2.COLOR_BGR2GRAY)
    img_lines = img_orj.copy()
    blur_gaus = cv2.GaussianBlur(gray,(3,3),0)
    kernel = np.ones((3,3), np.uint8)
    thresh_adaptive = cv2.adaptiveThreshold(blur_gaus,255,cv2.  ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,7,11)

    letter_img = letters(thresh_adaptive)

    cv2.imshow("letter_img", letter_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
