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


//Pedestrian structure: includes their location, depth, and enclosing rectangle.
struct Pedestrian {
	Point2i location; //Note, this is in the pixel coords of the image, not the depth map
	string depth_str;
	Rect rect;
};


//Initialize functions
static cv::Mat slMat2cvMat(sl::Mat& input);
static void findPedestrians(const HOGDescriptor &hog, cv::Mat& image_input, vector<Pedestrian> &peds);
static void findDepth(Pedestrian &ped, Size display_size, sl::Mat& depth_input);
static Point2i convertLocation(Point2i input_point, Size original_size, Size new_size);



/** Main Function overview-
	After necessary initializations, get the image and resize it.
	Every 4 loops, run pedestrian detection find depth of targets.
	Display that information on the image.
	Until information is updated, continue to display original info,
	even as the image itself changes. **/
int main(int argc, char **argv) {
	
	Camera zed;
	
	//Set camera parameters
	InitParameters init_params;
    init_params.camera_resolution = RESOLUTION_VGA;
    init_params.depth_mode = DEPTH_MODE_PERFORMANCE;
    init_params.camera_fps = 15;  //Lowest possible
    init_params.coordinate_units = UNIT_METER;
    
    //Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != SUCCESS) return 1;
    
    //Set runtime parameters.  Depth mapping is initially off.
    RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = SENSING_MODE_FILL;
    runtime_parameters.enable_depth = false;
    runtime_parameters.enable_point_cloud = false;
    
    //Create sl and cv matrix to house image.
    Resolution image_size = zed.getResolution();
    sl::Mat image_zed(image_size, sl::MAT_TYPE_8U_C4); 
    cv::Mat image_cv = slMat2cvMat(image_zed);
    
    //Create 2 more cv matrix: one to remove alpha channel, one to resize image
    cv::Mat image_cv_C3(image_size.width, image_size.height, CV_8UC3);
    Size display_size(720,404); //TODO: Can be tweaked for more accurate detection
    cv::Mat image_cv_resized(display_size, CV_8UC3);
    
    //Create sl matrix for depth
    sl::Mat depth_zed(image_size, sl::MAT_TYPE_32F_C1);
    
    //Initialize display window
    namedWindow("Image", WINDOW_NORMAL);
    
    //Code only for the Jetson: still not entirely sure what it does
    Camera::sticktoCPUCore(2);
    
    //Initialize pedestrian detection
    HOGDescriptor hog;
   	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    
    //Turn off unneeded processes (might not be necessary)
    zed.disableSpatialMapping();
    zed.disableTracking();
    
    //Track overall time of operation and number of loops run
    double time = (double) getTickCount();
    int num_loops = 0;
    
    //Only run pedestrian detector every 4 frames
    int frame_count = 1;
    
    //Vector of pedestrian information to be added to the image
    vector<Pedestrian> peds;
    
    //Loop until q is pressed
    char key = ' ';
    while (key != 'q') {
    	
    	//Measure time for each loop
    	double loop_time = (double) getTickCount();
    	
    	//If 4th frame, turn on depth mapping
    	if (frame_count == 4) {
    		runtime_parameters.enable_depth = true;
    		frame_count = 1;
    	} else frame_count++;
    	
    	//Grab and display processed image
    	if (zed.grab(runtime_parameters) == SUCCESS) {
    		zed.retrieveImage(image_zed, VIEW_LEFT);
    		
    		//remove alpha channel and resize
    		cvtColor(image_cv, image_cv_C3, CV_BGRA2BGR);
    		resize(image_cv_C3, image_cv_resized, display_size);
    		
    		//4th loop only, look for pedestrians
    		if (runtime_parameters.enable_depth) {
    			peds.clear(); //delete original data
    			
    			findPedestrians(hog, image_cv_resized, peds);
    			
    			//only retrieve depth info if pedestrians are detected
    			if (!peds.empty()) {
    				zed.retrieveMeasure(depth_zed, MEASURE_DEPTH);
    				
    				//Iterate through pedestrians and get depth
    				for (size_t i = 0; i < peds.size(); i++) {
    					findDepth(peds[i], display_size, depth_zed);
    					cout << i << ": " << peds[i].depth_str << endl;
    				}
    			}
    			runtime_parameters.enable_depth = false;
    		}
    		
    		//Regardless of loop number, add necessary information to image
    		for (size_t i = 0; i < peds.size(); i++) {
    			rectangle(image_cv_resized, peds[i].rect.tl(), peds[i].rect.br(), Scalar(0,255,0), 3);
    			putText(image_cv_resized, peds[i].depth_str, peds[i].location, FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
    		}
    		
    		imshow("Image", image_cv_resized);
    		
    	}
    	
    	//display loop time
    	loop_time = (double) getTickCount() - loop_time;
    	//cout << "Loop Time: " << loop_time/getTickFrequency() << endl;
    	
    	num_loops++;
    	
    	key = waitKey(10);
    }
    
    //print average loop time
    time = ((double) getTickCount() - time) / num_loops;
    cout << "Average Loop Time: " << (time/getTickFrequency()) << endl;
    
    zed.close();
    return 0;
}




/** Converts an sl matrix (from the ZED SDK) to an openCV matrix
	Taken directly from aforementioned github link
	Minor alteration to order of the cases**/
cv::Mat slMat2cvMat(sl::Mat& input) {
    //convert MAT_TYPE to CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
    	case sl::MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
        case sl::MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
        default: break;
    }

    // cv::Mat data requires a uchar* pointer. Therefore, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    //cv::Mat and sl::Mat will share the same memory pointer
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
}
    			


/** Locates pedestrians in a given image.
	If a pedestrian is found, a rectangle around them is created.
	Information is stored in a Pedestrian struct **/
static void findPedestrians(const HOGDescriptor &hog, cv::Mat& image_input, vector<Pedestrian> &peds) {
    
    //Create vector of rectangles for located pedestrians
    vector<Rect> found;
    
    hog.detectMultiScale(image_input, found, 0, Size(8,8), Size(8,8), 1.05, 2); //TODO: adjust values
    
    //check for smaller boxes occuring inside larger ones
    for (size_t i = 0; i < found.size(); i++) {
    	Rect r = found[i];

        size_t j;
        for (j = 0; j < found.size(); j++)
        	if (j != i && (r & found[j]) == r) break;

		//If all is well, add info to pedestrian vector
        if (j == found.size()) {
        	Pedestrian ped;
        	ped.rect = r;
        	
        	//Calculate location (center of the rectangle)
        	Point p = r.tl() + r.br();
        	p *= 0.5;
        	ped.location = p;
        	
        	peds.push_back(ped);
        }
    }
}



/** Takes location (in the resized frame) and finds the depth at the corresponding location **/
static void findDepth(Pedestrian &ped, Size display_size, sl::Mat& depth_input) {

	Size depth_size(depth_input.getWidth(), depth_input.getHeight());
	
	//Convert location to corresponding depth location
	Point2i depth_loc = convertLocation(ped.location, display_size, depth_size);

	//Get depth
	float depth_value;
	depth_input.getValue(depth_loc.x, depth_loc.y, &depth_value);
	
	//Convert to a string and add to ped
	stringstream stream;
	stream << fixed << setprecision(2) << depth_value;
	ped.depth_str = stream.str() + " m";
}



/**Takes a pixel location of an image and returns the corresponding location for a different size **/
static Point2i convertLocation(Point2i input_point, Size original_size, Size new_size) {
	
	Point2i output_point;
	
	//Convert point
	output_point.x = (int)input_point.x*((double)new_size.width/(double)original_size.width);
	output_point.y = (int)input_point.y*((double)new_size.height/(double)original_size.height);
	
	return output_point;
}
