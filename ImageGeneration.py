import cv2
import numpy as np

# Load the black and white doodle image
image = cv2.imread("D:\School\F23\ML\ML Project\Drawing Library\Apple\Apple3.png", cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur for noise reduction
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Detect lines using Hough Line Transform
lines = cv2.HoughLines(blurred_image, 1, np.pi / 180, threshold=100)

# Detect circles using Hough Circle Transform
circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0,
                           maxRadius=0)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # Draw lines on the image
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if circles is not None:
            circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
        radius = circle[2]

        # Draw circles on the image
        cv2.circle(image, center, radius, (0, 0, 255), 2)

        # Display the image with detected shapes
        cv2.imshow("Detected Shapes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
