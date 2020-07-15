#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

//OVERALL PROGRAM FLOW:
/* 
The problem is that regardless of foot sole capture with edge detection, bounding, etc., foot size requires a cm/pixel metric in order to be 
useful in determining real-life size. This can be accomplished in a variety of ways (camera movement with known displacement, geometry with extended 
object, knowledge of camera focal length, visual reference to known object length, etc.). The method chosen for this challenge is a reference object,
as the customer will have to complete the reference process themselves, so it needs to be simplistic. If this program were to be used to determine insole
size, customers would be asked to hold up a Toonie next to a picture of the bottom of their foot.

Essentially, a "calibration" will be performed with an object of known size (in this challenge, it is a Canadian Toonie), from which the cm/px ratio
can be computed and used to determine foot size. This still has some error as there is no guarantee that the Toonie/foot are parallel to the camera,
but fixing this entirely would be too complex and likely not very effective.

Overall Codeflow:
1.) Base image is converted to HSV, put through some low-pass filters, and thresholded to isolate skin-like pixels
2.) HSV image is passde through canny edge detector (some further inbuilt filters), to isolate contours
3.) Polygons are built off closed contours, largest closed polygon is recognized as footRectangle
4.) Base image is converted to grayscale and filtered to be hough circle transformed, in order to isolate circular Toonie
5.) Largest circular contour in picture is recognized as Toonie, cm/px ratio is calculated using known measurement
6.) Foot size is calculated using pixel measurements and known ratio, and displayed
*/

int main() {

    //pixels_per_metric (corresponds to cm)
    float pixels_per_metric;

    //Const Toonie radius (cm) - known and used for metric calculation 
    const float Toonie_RADIUS = 1.325;

	//TESTING IMAGE (1)
	Mat bottomTestImage = imread("soleTestWithReference.png");

    // HSV conversion + gaussian filtering to filter image based on skin color - more effective than morphological filtering
    GaussianBlur(bottomTestImage, bottomTestImage, Size(3, 3), 0, 0);
    Mat hsv; vector<Mat> chann; cvtColor(bottomTestImage, hsv, COLOR_RGB2HSV);
    split(hsv, chann); hsv = chann[1];
    threshold(hsv, hsv, 45, 255, THRESH_TOZERO);

    //canny edge detection (constants determined empirically)
    Mat cannyHSV; Canny(hsv, cannyHSV, 150, 225);

    // contour inititialization
    vector<vector<Point> > contours; vector<Vec3f> circles;
    findContours(cannyHSV, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<vector<Point> > contours_poly(contours.size()); 
    vector<Rect> boundRect(contours.size());

    // specific footRectangle, maxArea for iterative comparison
    int maxArea = 0; Rect footRect;

    // find largest, enclosed boundingRect polygon (will pertain to foot surface)
    for (size_t i = 0; i < contours.size(); i++) {
        
        //build polygon from found contour points
        approxPolyDP(contours[i], contours_poly[i], 3, true);
        boundRect[i] = boundingRect(contours_poly[i]);

        // sort for footRect while iterating 
        if ((boundRect[i].width * boundRect[i].height) > (footRect.width * footRect.height)) footRect = boundRect[i];
    }

    // grayscale conversion + blurring (with 3x kernel) for hough transformation
    Mat gray; 
    cvtColor(bottomTestImage, gray, COLOR_RGB2GRAY); GaussianBlur(gray, gray, Size(3, 3), 0, 0); int TooniePixelRadius = 0;

    // hough circle transform with empirical parameters
    // minimum distance/size such that no circular contour found on foot surface is valid - filtering + HSV should ensure that only Toonie is found
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1.5, 50, 150, 40, 0, 30);

    // cycling through found circular contours 
    Scalar color = Scalar(0, 128, 0);
    for (size_t i = 0; i < circles.size(); i++) {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        int radius = c[2];

        // sort for ToonieRadius while iterating
        if (radius > TooniePixelRadius) TooniePixelRadius = radius;
        circle(bottomTestImage, center, radius, color, 2, LINE_AA);
    }

    //PIXELS_PER_METRIC calculation - using known radius value to calculate picture scale 
    pixels_per_metric = Toonie_RADIUS / TooniePixelRadius;

    //image output
    rectangle(bottomTestImage, footRect.tl(), footRect.br(), color, 2);
    putText(hsv, "HSV-Filtered Image", Point(10, hsv.rows / 10), FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(255, 255, 255), 2);
    imshow("ImageOutput", hsv);
    waitKey(6000);
    putText(cannyHSV, "Canny Edge Detection", Point(10, hsv.rows / 10), FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(255, 255, 255), 2);
    imshow("ImageOutput", cannyHSV);
    waitKey(4000);
    putText(bottomTestImage, "Detected Foot + Toonie Contours", Point(10, hsv.rows / 10), FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(0, 0, 0), 2);
    imshow("ImageOutput", bottomTestImage);
    waitKey(4000);

    //foot size + pixel metric console outputs
    std::cout << "Toonie MEASUREMENT OUTPUTS \n";
    std::cout << "********************************** \n";
    std::cout << "Radius in image (pixels) " << TooniePixelRadius << std::endl;
    std::cout << "Radius in real-life (known constant) - " << Toonie_RADIUS << std::endl;
    std::cout << "Centimetres-Per-Pixel - " << pixels_per_metric << std::endl;
    std::cout << "\nFOOT MEASUREMENT OUTPUTS \n";
    std::cout << "********************************** \n";
    std::cout << "Foot Length (pixels) - " << footRect.height << std::endl;
    std::cout << "Foot Length (cm) - " << footRect.height * pixels_per_metric << std::endl;
    std::cout << "Foot Width (pixels) - " << footRect.width << std::endl;
    std::cout << "Foot Width(cm) - " << footRect.width * pixels_per_metric << std::endl;
    
    //waitKey for highGUI execution
	waitKey(0);
}