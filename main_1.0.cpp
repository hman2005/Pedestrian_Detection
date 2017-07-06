/** Pedestrian detection using input from the ZED camera.
		Takes images from left lens of ZED and runs opencv algorithm to locate people.
		Then uses the depth map to calculate and display person's distance from camera.
		
		slMat2cvMat function and a good bit of the main function copied from official zed github: https://github.com/stereolabs/zed-opencv
		
				
	Harrison Fay **/

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Core.hpp>
#include <sl/defines.hpp>
#include <opencv2/objdetect.hpp>

using namespace sl;
using namespace cv;
using namespace std;

cv::Mat slMat2cvMat(sl::Mat& input);
static void findPedestrian(const HOGDescriptor &hog, cv::Mat& input, cv::Mat& depthInput);

int main(int argc, char **argv) {

    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION_HD720;
    init_params.depth_mode = DEPTH_MODE_PERFORMANCE;
    init_params.camera_fps = 4;
    init_params.coordinate_units = UNIT_METER;

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != SUCCESS)
        return 1;

    // Set runtime parameters after opening the camera
    RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = SENSING_MODE_FILL; // Use STANDARD sensing mode

    // Create sl and cv Mat to get ZED left image and depth image and depth information
    // Best way of sharing sl::Mat and cv::Mat :
    // Create a sl::Mat and then construct a cv::Mat using the ptr to sl::Mat data.
    Resolution image_size = zed.getResolution();
    cv::Size cvImageSize(image_size.width, image_size.height);
    
    sl::Mat image_zed(image_size,sl::MAT_TYPE_8U_C4); // Create a sl::Mat to handle Left image
	cv::Mat image_ocv = slMat2cvMat(image_zed);
	cv::Mat image_ocv3(cvImageSize, CV_8UC3);
	
	sl::Mat depth_image_zed(image_size, MAT_TYPE_8U_C4);
	cv::Mat depth_image_ocv = slMat2cvMat(depth_image_zed);
	
	sl::Mat depth_zed(image_size, MAT_TYPE_32F_C1);
	cv:: Mat depth_ocv = slMat2cvMat(depth_zed);
	

    // Create OpenCV images to display (lower resolution to fit the screen)
    cv::Size displaySize(720, 404);
    cv::Mat image_ocv_display(displaySize, CV_8UC3);
    cv::Mat depth_image_ocv_display(displaySize, CV_8UC4);
    cv::Mat depth_ocv_display(displaySize, CV_32FC1); 

    // Give a name to OpenCV Windows
    namedWindow("Image", cv::WINDOW_NORMAL);
    namedWindow("Depth", cv::WINDOW_NORMAL);

    // Jetson only. Execute the calling thread on 2nd core
    Camera::sticktoCPUCore(2);
    
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    // Loop until 'q' is pressed
    char key = ' ';
    while (key != 'q') {
    
    	//Measure time required for each loop
    	double t = (double) getTickCount();

        // Grab and display image and depth 
        if (zed.grab(runtime_parameters) == SUCCESS) {

            zed.retrieveImage(image_zed, VIEW_LEFT); // Retrieve the left image
            zed.retrieveImage(depth_image_zed, VIEW_DEPTH); //Retrieve the depth view (image)
            zed.retrieveMeasure(depth_zed, MEASURE_DEPTH); //Retrieve depth info
            
            //Convert image to 8UC3 (necessary for HOG)
            cvtColor(image_ocv, image_ocv3, CV_BGRA2BGR);            
			
            // Resize and display with OpenCV
            resize(image_ocv3, image_ocv_display, displaySize);
            resize(depth_ocv, depth_ocv_display, displaySize);
            //Locate pedestrian in image
            findPedestrian(hog, image_ocv_display, depth_ocv_display);
            imshow("Image", image_ocv_display);
            
            resize(depth_image_ocv, depth_image_ocv_display, displaySize);
            imshow("Depth", depth_image_ocv_display);
            
            t = (double) getTickCount() - t;
            cout << (t*1000./cv::getTickFrequency()) << endl;
            key = waitKey(10);
        }
    }

    zed.close();
    return 0;
}

/**Converts an sl matrix (from the ZED SDK) to an openCV matrix**/
cv::Mat slMat2cvMat(sl::Mat& input) {
	//convert MAT_TYPE to CV_TYPE
	int cv_type = -1;
	switch (input.getDataType()) {
		case sl::MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
		case sl::MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
		case sl::MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
		case sl::MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
		case sl::MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
		case sl::MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
		case sl::MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
		case sl::MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
		default: break;
	}

	// cv::Mat data requires a uchar* pointer. Therefore, we get the uchar1 pointer from sl::Mat (getPtr<T>())
	//cv::Mat and sl::Mat will share the same memory pointer
	return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
}

/** Takes an image (in the form of an openCV matrix) and 
	attempts to find pedestrians.  If pedestrian is found, 
	a box is drawn around him/her/whatever is the appropriate pronoun
	Additionally, uses depth map to return distance to pedestrian **/
static void findPedestrian(const HOGDescriptor &hog, cv::Mat& input, cv::Mat& depthInput) {
	//double t = (double) getTickCount();
	
	vector<Rect> found, found_filtered; 
	//Detect pedestrians and write those detections to found.
	hog.detectMultiScale(input, found, 0, Size(4,4), Size(8,8), 1.05, 2);
	
	for (size_t i = 0; i < found.size(); i++) {
		Rect r = found[i];
		
		size_t j;
		//Prevent smaller boxed from occuring inside larger ones.
		for (j = 0; j<found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		
		//place r in found_filtered
		if (j == found.size())
			found_filtered.push_back(r);
		}
	
	string distance; //initialize distance to be displayed
	for (size_t i = 0; i < found_filtered.size(); i++) {
		Rect r = found_filtered[i];
		
		//Apparently the default makes the rectangle a bit too large,
		//So here we shrink it down a bit.
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		rectangle(input, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
		
		int centerX = r.x+r.width/2;
		int centerY = r.y+r.height/2;

		//NOTE: make finding depth a separate function once this is working
		//Take average of data points around center (ignore those without a value)
		//May not be necessary to do.
		float distance = 0.0;
		int sample_number = 0;
		for (int x = -2; x <= 2; x++) {
			for (int y = -2; y<= 2; y++) {
				float temp_distance = depthInput.at<float>(centerX+x, centerY+y);
				//make sure temp_distance is valid before adding to distance
				if (!isnan(temp_distance) && temp_distance > 0.5 && temp_distance < 20.0) {
					distance += temp_distance;
					sample_number++;
				}
			}
		}
		//if sample_number is still zero, just print 0.  Otherwise, average distance by sample_number
		if (sample_number != 0) distance = distance/sample_number;
		
		//Get depth at center of rectangle and display
		string distance_string = to_string(distance);
		//cout << distance_string << endl;
		putText(input, distance_string, cvPoint(centerX,centerY), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
	}
	//t = (double) getTickCount() - t;
	//cout << "HOG" << (t*1000./cv::getTickFrequency()) << endl;
} 

