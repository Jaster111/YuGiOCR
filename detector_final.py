import cv2
import os
import numpy as np
from utils import four_point_transform
from deskew import determine_skew
from skimage.transform import rotate
import json
import difflib
from PIL import Image
import argparse
import pytesseract


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="path to image")
    parser.add_argument("--tesseract", help="path to tesseract.exe")
    parser.add_argument("--visualize", help="boolean, visualize intermediate steps", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.image) or not args.image.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        print("Not a valid image path!")
        exit(0)

    if not os.path.exists(args.tesseract) or not args.tesseract.lower().endswith(("tesseract.exe")):
        print("Not a valid tesseract executable!")
        exit(0)

    pytesseract.pytesseract.tesseract_cmd = args.tesseract

    f = open("./cardinfo.php", "rb")
    card_data = json.loads(f.read())
    f.close()

    card_names = [i["name"].upper() for i in card_data["data"]]

    img = cv2.imread(args.image)

    #MAIN IMAGE PREPROCESSING
    #Resizing to 800x800
    ratio = img.shape[0] / 800.
    img_resized = cv2.resize(img, (800, 800))
    coef_y = img.shape[0] / img_resized.shape[0]
    coef_x = img.shape[1] / img_resized.shape[1]
    
    #grayscale -> bluring -> canny thresholding -> dilation
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_img, (11,11), 0)
    thresh = cv2.Canny(blur, 50, 100)
    dilated = cv2.dilate(thresh, np.ones((11,11), dtype=np.int8))

    if args.visualize:
        cv2.imshow("dilated", dilated)
        cv2.waitKey(0)

    #CONTOUR EXTRACTION
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #FINDING 4-POINT CONTOURS BECAUSE THOSE MOST OFTEN RESEMBLE A RECTANGLE / CARD SHAPE
    tbd = list()
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) == 4:
            tbd.append(approx)


    new_contours = list()

    #ITERATING THROUGH 4-POINT CONTOURS AND FINDING THE CARDS AMONG THEM
    for c in tbd:
        x,y,w,h = cv2.boundingRect(c)

        if (h > 100 and w > 100):
                
            warped = four_point_transform(img_resized, c.reshape((4,2)))
            warped_w = warped.shape[0]
            warped_h = warped.shape[1]
            print("Warped w: {}, Warped h: {}, ratio: {}, scaled ratio: {}".format(warped_w, warped_h, warped_h/warped_w, (warped_h*coef_y) / (warped_w*coef_x)))
            if (1.15 < (warped_h*coef_y) / (warped_w*coef_x) < 1.4):
                if args.visualize:
                    cv2.imshow("warped", warped)
                    cv2.waitKey(0)

                #DETERMINE ANGLE FOR DESKEWING
                angle = determine_skew(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))

                if angle != 0 and angle is not None:
                    warped = rotate(warped, angle, resize=True)

                #CROP THE TEXT BOX APPROXIMATELLY AND GET THE TEXT
                cropped = warped[int(warped.shape[1]//16):int(warped.shape[1]//7.7), int(warped.shape[0]*0.05):int(warped.shape[0]*0.73)]
                text_roi = cv2.resize(cropped, (cropped.shape[1]*3,cropped.shape[0]*3))

                if args.visualize:
                    cv2.imshow("extracted name", text_roi)
                    cv2.waitKey(0)

                text_roi = Image.fromarray((text_roi * 255).astype(np.uint8))

                query = pytesseract.image_to_string(text_roi, config="--psm 7")

                #IF TEXT HAS BEEN FOUND, MEMORIZE THE CONTOUR AND THE TEXT
                if query:
                    new_contours.append([c, query])
            


    #RESIZING THE CONTOURS BACK TO THE ORIGINAL IMAGE SIZE

    #DRAWING THE BOUNDING BOXES AND CORRESPONDING TEXT
    for c, query in new_contours:
        c[:, :, 0] = c[:, :, 0] * coef_x
        c[:, :, 1] = c[:, :,  1] * coef_y

        
        x,y,w,h = cv2.boundingRect(c)
        
        try:
            card_text = difflib.get_close_matches(query.upper(), card_names, n=1)[0]
            if card_text:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 6)
                (w, h), _ = cv2.getTextSize(card_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                cv2.rectangle(img, (x, y - h - 5), (x + w + 5, y), (0,0,0), -1)
                cv2.putText(img, card_text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
        except:
            continue

    cv2.imshow("img", img)
    cv2.imshow("img_res", cv2.resize(img, (500,500)))
    cv2.waitKey(0)