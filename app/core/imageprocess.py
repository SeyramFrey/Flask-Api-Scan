from PIL import Image
import cv2
import pytesseract

imagepath="C:/Users/gseyr/Pictures/ticket.jpg"

im = Image.open(imagepath)

img = cv2.imread(imagepath)
#im.rotate(45).show()
#For saving the image im.save("../core/enzo.jpg")

###Inverser les couleurs de l'image :
inverted_img = cv2.bitwise_not(im)

###Binarization(Convertir en blanc - Noir)

def grayscale(image):
    #Retourne un gray scale image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = grayscale(img)

#Cette ligne enregistre l'image
cv2.imwrite("../core/ticket.jpg")

#Convertir en noir et blanc
#En changeant les valeurs des nombres ci dessous, on ajuste la netteté de l'image
thresh, im_bw = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite("..core/bw_image.jpg", im_bw)