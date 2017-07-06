/** Pedestrian detection using input from the ZED camera.
	Takes images from left lens of ZED and runs opencv algorithm to locate pedestrian.
	Then uses the depth map to calculate and display person's distance from camera.
	
	slMat2cvMat function and a good bit of the main function copied from official zed github:
    	https://github.com/stereolabs/zed-opencv
    	
    Harrison Fay **/
    
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Core.hpp>
#include <sl/defines.hpp>
#include <opencv2/objdetect.hpp>

using namespace sl;
using namespace cv;
using namespace std;

//Initialize functions
static cv::Mat slMat2cvMat(sl::Mat& input);
static vector<Point2i> findPedestrian(const HOGDescriptor &hog, cv::Mat& image_input);
static void findDepth(Point2i location, cv::Mat& image_input, sl::Mat& depth_input);
static Point2i convertLocation(Point2i input_point, size_t width_orig, size_t height_orig, size_t width_new, size_t height_new);

/** Main Function Overview:
	After intializations, get image, resize, and run pedestrian detection algorithm.
	For each pedestrian located, use the depth map to find the distance from the camera.
	Use (u,v) coords to transition between sizes of image and depth map.
	Display the information on the image **/
int main(int argc, char **argv) {
	//Create the ZED object
	Camera zed;
	
	//Set configuration parameters
	InitParameters init_params;
    init_params.camera_resolution = RESOLUTION_VGA; //TODO: Try out VGA
    init_params.depth_mode = DEPTH_MODE_PERFORMANCE; //TODO: Try out MIDDLE
    init_params.camera_fps = 15;  //Lowest possible
    init_params.coordinate_units = UNIT_METER;
    
    //Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != SUCCESS) return 1;
    
    //Set runtime parameters (only after opening camera)
    RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = SENSING_MODE_FILL; //Not STANDARD: is faster, but map has big holes
    
    //Find image resolution (to properly size matrix)
    Resolution image_size = zed.getResolution();
    
    //Create sl and cv matrix for image (cv so it can be resized and displayed)
    sl::Mat image_zed(image_size, sl::MAT_TYPE_8U_C4); //Can this be C3 instead of C4? answer: nope
    cv::Mat image_cv = slMat2cvMat(image_zed);
    
    //Create cv matrix that removes 4th channel from image_cv (necessary to run HOG)
    cv::Mat image_cv_C3(image_size.width, image_size.height, CV_8UC3);
    
    //Create sl matrix for depth info
    sl::Mat depth_zed(image_size, sl::MAT_TYPE_32F_C1);
    
    //For testing purposes, also make sl and cv matrix to display depth map
    //sl::Mat depth_image_zed(image_size, MAT_TYPE_8U_C4);
    //cv::Mat depth_image_cv = slMat2cvMat(depth_image_zed);
    
    // Create sized-down cv matrix
    Size displaySize(720,404); //TODO: Tweak sizing for more accurate detection
    cv::Mat image_cv_resized(displaySize, CV_8UC3);
   	
   	//Initialize cv windows
   	namedWindow("Image", WINDOW_NORMAL);
   	//namedWindow("Depth", WINDOW_NORMAL);
   	
   	//Special code for the Jetson: executes the calling thread on 2nd core
   	Camera::sticktoCPUCore(2);
   	
   	//Initialize pedestrian detection
   	HOGDescriptor hog;
   	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
   	
   	//Turn off unneeded applications
   	zed.disableSpatialMapping();
   	zed.disableTracking();
   	
   	//Keep track of overall time of operation and the number of loops run
   	double time = (double) getTickCount();
   	int num_loops = 0;
   	
   	//Actual process begins here: loop until 'q' is pressed.
   	char key = ' ';
   	while (key != 'q') {
   	
   		//For testing purposes, measure time required for each loop.
   		double t = (double) getTickCount();
   		
   		//Grab and display processed image
   		if (zed.grab(runtime_parameters) == SUCCESS) {
   			
   			//retrieve information
   			zed.retrieveImage(image_zed, VIEW_LEFT);
            //zed.retrieveImage(depth_image_zed, VIEW_DEPTH);
            
            //Convert image_cv to C3
            cvtColor(image_cv, image_cv_C3, CV_BGRA2BGR);
            
            //Resize image and run pedestrian detection.
            //This also draws rectangles around pedestrians in the image
            resize(image_cv_C3, image_cv_resized, displaySize);
            vector<Point2i> centers = findPedestrian(hog, image_cv_resized);
            
            //cout << "Size of centers: " << centers.size() << endl;
            
            //Only retrive depth info if there are actually people in the image
            if (!centers.empty()) {
            	zed.retrieveMeasure(depth_zed, MEASURE_DEPTH);
   		   		//Find depth for each pedestrian and add info to image
   		   		for (size_t i = 0; i < centers.size(); i++) {
   		   			//cout << "Center: " << centers[i].x << " " << centers[i].y << endl;
   		   			findDepth(centers[i], image_cv_resized, depth_zed);
   		   		}
   		   	}
   		   	
   		   	//Display image
   		   	imshow("Image", image_cv_resized);
   		   	//imshow("Depth", depth_image_cv);
   		   	
   		   	//continuation of testing for loop time
   		   	t = (double) getTickCount() - t;
   		   	cout <<(t*1000./getTickFrequency()) << endl;
   		   	key = waitKey(10);
   		}
   		num_loops++;
   	}
   	
   	//print average duration of a loop
   	time = ((double) getTickCount() - time) / num_loops;
   	cout << "Average loop time: " << (time/getTickFrequency()) << endl;
   	
   	zed.close();
   	return 0;
}

/** Converts an sl matrix (from the ZED SDK) to an openCV matrix
	Taken directly from aforementioned github link**/
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

/** Takes image (cv matrix) and attempts to find pedestrians.
	If pedestrian is found, a rectangle is drawn.\
	Return the center of each rectangle, converted to (u,v) coordinates **/
static vector<Point2i> findPedestrian(const HOGDescriptor &hog, cv::Mat& image_input) {
	
	//Create vector of rectangles for located pedestrians (found) and another to check overlap (found_filtered)
	vector<Rect> found, found_filtered;
	//Run algorithm. TODO: tweak values
	hog.detectMultiScale(image_input, found, 0, Size(8,8), Size(8,8), 1.05, 2);
	
	
	for (size_t i = 0; i < found.size(); i++) {
    	Rect r = found[i];

        size_t j;
        //Prevent smaller boxed from occuring inside larger ones.
        for (j = 0; j<found.size(); j++)
        	if (j != i && (r & found[j]) == r) break;

		//place r in found_filtered
        if (j == found.size()) found_filtered.push_back(r);
    }
    
    //Write rectangles to image and output their centers
    vector<Point2i> centers;
    for (size_t i = 0; i < found_filtered.size(); i++) {
    	Rect r = found_filtered[i];
    	
    	//Get top-left and bottom-right points of r
    	Point tl = r.tl();
    	Point br = r.br();
    	
    	//Draw rectangle (in green)
    	rectangle(image_input, tl, br, Scalar(0,255,0), 3);
    	
    	//Average tl and br to get the center.
    	Point p = tl + br;
    	p *= 0.5;
    	//cout << "Rectangle Center: " << p.x << " " << p.y << endl;
    	centers.push_back(p);
    }
    return centers;
}

/** Takes location (in resized frame), converts to size of depth map,
	and finds depth at that location.  Draws information to image. **/
static void findDepth(Point2i location, cv::Mat& image_input, sl::Mat& depth_input) {
	//cout << "Original Location: " << location.x << " " << location.y << endl;

    //Resize location so that it applies to the depth info
    Size image_size = image_input.size();
    Resolution depth_res = depth_input.getResolution();
    Point2i new_loc = convertLocation(location, image_size.width, image_size.height, depth_res.width, depth_res.height);
    
    //cout << "New Location: " << new_loc.x << " " << new_loc.y << endl;
    
    //get depth
    float depth_value = 0.0;
    depth_input.getValue(new_loc.x, new_loc.y, &depth_value);
    
    //Draw
    stringstream stream;
    stream << fixed << setprecision(1) << depth_value;
    string depth_string = stream.str() + " m";
    putText(image_input, depth_string, location, FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
}

/** Takes a pixel location of an image and returns the corresponding pixel in a different-sized image **/
static Point2i convertLocation(Point2i input_point, size_t width_orig, size_t height_orig, size_t width_new, size_t height_new) {

	Point2i output_point;
	
	//Convert point.  Be careful of errors due to typing!
	output_point.x = (int)input_point.x*((double)width_new/(double)width_orig);
	output_point.y = (int)input_point.y*((double)height_new/(double)height_orig);
	
	return output_point;
}
	
