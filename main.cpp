#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <cstddef>
#include <vector>
#include <iostream>
#include <cstdlib>

using namespace cv;
using namespace std;

const int number_frames_average = 2;
const float average_weight = 0.013;

class Obj {
    public:
        cv::Point currentCentroid;
        cv::Point previousCentroid;
        void set_values (cv::Point,cv::Point);    
};

void Obj::set_values (cv::Point p1, cv::Point p2) {
  currentCentroid = p1;
  previousCentroid = p2;
}

bool checkPointIsInsideImage (cv::Point p, Mat image){
    float rows = image.rows;
    float cols = image.cols;
    return (p.x < rows && p.x >= 0 && p.y < cols && p.y >=0);
}

vector<Point> getCentroid(Mat gray){
    // detect edges using canny
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Canny( gray, canny_output, 50, 150, 3 );

    // find contours
    findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    // get the moments
    vector<Moments> mu(contours.size());
    for( int i = 0; i<contours.size(); i++ ){
        mu[i] = moments( contours[i], false);
    }

    // get the centroid of figures.
    vector<Point2f> mc(contours.size());
    for( int i = 0; i<contours.size(); i++){
        mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00);
    }
    /* Draw contours and add the centroid to the centroid list if the minimum size
     * of the figure and if it is within the limits of the image.*/
    vector<Point> centroid_list;
    Mat drawing(canny_output.size(), CV_8UC3, Scalar(255,255,255));
    for( int i = 0; i<contours.size(); i++ ){
        Scalar color = Scalar(150,120,50);
        if (contours[i].size() > 30 && checkPointIsInsideImage(mc[i], canny_output)){
            drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
            circle( drawing, mc[i], 4, color, -1, 8, 0 );
            centroid_list.push_back(mc[i]);
        }
    }
    // show the image with the regions centroids
    namedWindow( "Centroids", WINDOW_AUTOSIZE );
    imshow( "Centroids", drawing );    
    return centroid_list;
}

int getPreviousIterator(int iterator, int sizeOfArray){
    if (iterator == 0){
        return sizeOfArray-1;
    }else{
        return iterator-1;
    }
}

// get the average of the images that is their background
void averageImage(cv::Mat average_frames[number_frames_average], int iterator){
    int height = average_frames[iterator].rows;
    int width = average_frames[iterator].cols;    
    int previous_iterator = getPreviousIterator(iterator, number_frames_average);
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            average_frames[iterator].at<uchar>(i,j) = average_frames[iterator].at<uchar>(i,j)*average_weight + (1-average_weight)*average_frames[previous_iterator].at<uchar>(i,j);
        }
    }
}

// obtains the subtraction of the current frame with the middle image frame (background).
void subtractImage(cv::Mat frame, cv::Mat average_frames[number_frames_average], cv::Mat subtracted_image, int iterator){
    int height = frame.rows;
    int width = frame.cols;
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            subtracted_image.at<uchar>(i,j) = (frame.at<uchar>(i,j) - average_frames[iterator].at<uchar>(i,j));
        }
    }    
}

void thresholdImage(cv::Mat image, cv::Mat binary_image){
    int height = image.rows;
    int width = image.cols;
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            if (image.at<uchar>(i,j) > 150){
                binary_image.at<uchar>(i,j) = 255;
            }
            else{
                binary_image.at<uchar>(i,j) = 0;
            }
        }
    }
}

void drawCentroidObjects(cv::Mat image, vector<Obj> objects_list){
    for (auto &obj : objects_list){
        circle( image, obj.currentCentroid, 4, Scalar(167,151,0), -1, 8, 0 );
        circle( image, obj.previousCentroid, 4, Scalar(200,50,30), -1, 8, 0 );
        cv::putText(image,format("(%d,%d)", obj.currentCentroid.x, obj.currentCentroid.y),cv::Point(obj.currentCentroid),cv::FONT_HERSHEY_DUPLEX,0.4,cv::Scalar(0,100,0),1,false);
        cv::putText(image,format("(%d,%d)", obj.currentCentroid.x, obj.currentCentroid.y),cv::Point(obj.previousCentroid),cv::FONT_HERSHEY_DUPLEX,0.4,cv::Scalar(0,0,100),1,false);
    }
}

/* Lists the centroides of the objects detected in the previous frame to the current one, returning a list of objects that contain them.
 * Conditions to be able to relate the centroid of the previous frame to the current one: white tint, be the centroid closer and not
 * more than 40 distances.*/
vector<Obj> putCentroidObjects(vector<vector<Point>> centroid_history, cv::Mat frames[number_frames_average], int iterator){
    vector<Obj> objects_list;
    Point a;
    Point b;
    int previous_iterator = getPreviousIterator(iterator, number_frames_average);
    for (int k=0; k<centroid_history[iterator].size(); k++){
        double shortestDistance = 1000000.0;
        a = centroid_history[iterator][k];
        Obj obj;
        for (int l=0; l< centroid_history[previous_iterator].size();l++){
            b = centroid_history[previous_iterator][l];
            if (frames[iterator].at<uchar>(a) == frames[previous_iterator].at<uchar>(b)){
                double distance = cv::norm(a-b);
                if(distance < shortestDistance && distance < 40){
                    shortestDistance = distance;
                    obj.set_values(a,b);
                }
            }
        }
        if(objects_list.empty()){
            objects_list.push_back(obj);
        }else{
            if (objects_list.back().currentCentroid != obj.currentCentroid){
                objects_list.push_back(obj);
            }
        }
    }
    return objects_list;
}

void countObjects(cv::Mat image, vector<Obj> objects_list, int &num_persons){
    int height = image.rows;
    for (int k=0; k<objects_list.size(); k++){
        if (objects_list[k].currentCentroid.y <= (height/2) && objects_list[k].previousCentroid.y > (height/2)){
            num_persons++;
        }else if(objects_list[k].currentCentroid.y >= (height/2) && objects_list[k].previousCentroid.y < (height/2)){
            num_persons--;
        }
    }
}

int main( int argc, char** argv ) {
   int num_persons = 0;
  // Create a VideoCapture object and open the input file  
  VideoCapture cap("video.mp4");
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  // array with the 
  Mat average_frames[number_frames_average];
  Mat frames[number_frames_average];
  vector<vector<cv::Point>> centroid_history;

  int iterator = 0;
  int countFrames = 0;
  bool arrayFinalFull = false;  
  // image frame
  Mat frame;
  Mat colorframe;

  while(1){

    // Capture frame-by-frame
    cap >> colorframe;

    // If the frame is empty, break immediately
    if (colorframe.empty()){
        cout <<" empty" << endl;
        break;
    }

    cvtColor(colorframe, frame, CV_BGR2GRAY);
    if (frame.empty()){
        cout <<" empty" << endl;
        break;
    }
    
    cv::GaussianBlur(frame, frame, cv::Size(5, 5), 0);

    frames[iterator] = frame.clone();
    average_frames[iterator] = frame.clone();    

    Mat binary_image = frame.clone();
    Mat result = colorframe.clone();
    Mat subtracted_image = frame.clone();

    if (arrayFinalFull) {

        averageImage(average_frames, iterator);
        subtractImage(frame, average_frames, subtracted_image, iterator);
        cv::GaussianBlur(binary_image, binary_image, cv::Size(5, 5), 0);

        thresholdImage(subtracted_image, binary_image);

        cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));

        cv::dilate(binary_image, binary_image, structuringElement7x7);
        cv::dilate(binary_image, binary_image, structuringElement5x5);
        cv::erode(binary_image, binary_image, structuringElement5x5);

        std::vector<std::vector<cv::Point> > contours;
        Mat binary_image_copy = binary_image.clone();

        cv::findContours(binary_image_copy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat image(binary_image.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));
        cv::drawContours(image, contours, -1, cv::Scalar(255.0, 255.0, 255.0), -1);
        cv::imshow("imgContours", image);
        frames[iterator] = image.clone();

        vector<Point> centroid_list = getCentroid(image.clone());
        /* "number_frames_average+2" means that the firsts two positions of the vector are empty,
         *  that is, the list can not be added to the end of the vector. Otherwise, we will just
         *  replace the existing lists in the array*/
        if (countFrames < number_frames_average+2){
            centroid_history.push_back(centroid_list);
        }else{
            centroid_history[iterator]=centroid_list;
        }

        vector<Obj> objects_list;
        if (countFrames > 2){
            objects_list = putCentroidObjects(centroid_history, frames, iterator);
            drawCentroidObjects(result, objects_list);

            countObjects(result, objects_list, num_persons);

        }

        imshow( "Binary image", binary_image);
        imshow( "Average", average_frames[iterator]);
        imshow( "Frame", frame);

    }
    cv::putText(result,"Flow: " + std::to_string(num_persons),cv::Point(20,60),cv::FONT_HERSHEY_DUPLEX,0.7,cv::Scalar(255,20,150),1,false);
    cv::putText(result,"Frames: " + std::to_string(countFrames),cv::Point(20,30),cv::FONT_HERSHEY_DUPLEX,0.7,cv::Scalar(255,0,0),1,false);
    imshow( "Resultado", result);

    iterator++;
    if (iterator == number_frames_average){
        iterator = 0;
        if (not arrayFinalFull){
            arrayFinalFull = true;
        }

    }    
    countFrames++;
    waitKey(0);

    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);    
    if(c==27)
      break;
  }
  // When everything done, release the video capture object
  cap.release();
  // Closes all the frames
  destroyAllWindows();
  return 0;
}
