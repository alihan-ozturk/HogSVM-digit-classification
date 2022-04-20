#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <time.h>
using namespace cv;
using namespace cv::ml;
using namespace std;
vector< float > get_svm_detector( const Ptr< SVM >& svm );
void convert_to_ml( const std::vector< Mat > & train_samples, Mat& trainData );
void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages );
void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size );
void computeHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst, bool use_flip );
void test_trained_detector( String obj_det_filename, String test_dir );
vector< float > get_svm_detector( const Ptr< SVM >& svm )
{
    // get the support vectors
    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction( 0, alpha, svidx );
    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );
    vector< float > hog_detector( sv.cols + 1 );
    memcpy( &hog_detector[0], sv.ptr(), sv.cols*sizeof( hog_detector[0] ) );
    hog_detector[sv.cols] = (float)-rho;
    return hog_detector;
}

void convert_to_ml( const vector< Mat > & train_samples, Mat& trainData )
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    Mat tmp( 1, cols, CV_32FC1 ); //< used for transposition if needed
    trainData = Mat( rows, cols, CV_32FC1 );
    for( size_t i = 0 ; i < train_samples.size(); ++i )
    {
        CV_Assert( train_samples[i].cols == 1 || train_samples[i].rows == 1 );
        if( train_samples[i].cols == 1 )
        {
            transpose( train_samples[i], tmp );
            tmp.copyTo( trainData.row( (int)i ) );
        }
        else if( train_samples[i].rows == 1 )
        {
            train_samples[i].copyTo( trainData.row( (int)i ) );
        }
    }
}
void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages = false )
{
    vector< String > files;
    glob( dirname, files );
    for ( size_t i = 0; i < files.size(); ++i )
    {
        Mat img = imread( files[i] ); // load the image
        if ( img.empty() )
        {
            cout << files[i] << " is invalid!" << endl; // invalid image, skip it.
            continue;
        }
        if ( showImages )
        {
            imshow( "image", img );
            waitKey( 1 );
        }
        img_lst.push_back( img );
    }
}
void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size )
{
    Rect box;
    box.width = size.width;
    box.height = size.height;
    srand( (unsigned int)time( NULL ) );
    for ( size_t i = 0; i < full_neg_lst.size(); i++ )
        if ( full_neg_lst[i].cols > box.width && full_neg_lst[i].rows > box.height )
        {
            box.x = rand() % ( full_neg_lst[i].cols - box.width );
            box.y = rand() % ( full_neg_lst[i].rows - box.height );
            Mat roi = full_neg_lst[i]( box );
            neg_lst.push_back( roi.clone() );
        }
}
void computeHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst, bool use_flip )
{
    HOGDescriptor hog;
    hog.winSize = wsize;
    Mat gray;
    vector< float > descriptors;
    for( size_t i = 0 ; i < img_lst.size(); i++ )
    {
        if ( img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height )
        {
            Rect r = Rect(( img_lst[i].cols - wsize.width ) / 2,
                          ( img_lst[i].rows - wsize.height ) / 2,
                          wsize.width,
                          wsize.height);
            cvtColor( img_lst[i](r), gray, COLOR_BGR2GRAY );
            hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );
            gradient_lst.push_back( Mat( descriptors ).clone() );
            if ( use_flip )
            {
                flip( gray, gray, 1 );
                hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );
                gradient_lst.push_back( Mat( descriptors ).clone() );
            }
        }
    }
}
void test_trained_detector( String obj_det_filename, String test_dir)
{
    cout << "Testing trained detector..." << endl;
    HOGDescriptor hog;
    hog.load( obj_det_filename );
    vector< String > files;
    glob( test_dir, files );
    int delay = 1;
    obj_det_filename = "testing " + obj_det_filename;
    namedWindow( obj_det_filename, WINDOW_NORMAL );
    for( size_t i=0;; i++ )
    {
        Mat img;
        
        img = imread( files[i] );

        if ( img.empty() )
        {
            return;
        }
        vector< Rect > detections;
        vector< double > foundWeights;
        hog.detectMultiScale( img, detections, foundWeights );
        if(detections.size()>0)
        {
        for ( size_t j = 0; j < detections.size(); j++ )
        {
            Scalar color = Scalar( 0, 0, 255 );
            rectangle( img, detections[j], color, img.cols / 400 + 1 );
            imshow( obj_det_filename, img );
            if( waitKey( 0 ) == 27 )
            {
                return;
            }
        }
        }
        else
        {
        imshow( obj_det_filename, img );
        if( waitKey( delay ) == 27 )
        {
            return;
        }
        }
        
    }
}
int main( int argc, char** argv )
{
    String pos_dir = "C:/Users/Alihan/Desktop/openCValg/dataset/100_SIGN";
    String neg_dir = "C:/Users/Alihan/Desktop/openCValg/dataset/NO";
    String test_dir = "C:/Users/Alihan/Desktop/trafficSingDet-main/images/val";
    String obj_det_filename = "trained.yml";
    int detector_width = 32;
    int detector_height = 32;
    bool test_detector = false;
    bool visualization = false;
    bool flip_samples = true;

    vector< Mat > pos_lst, full_neg_lst, neg_lst, gradient_lst;
    vector< int > labels;
    cout << "Positive images are being loaded..." ;
    load_images( pos_dir, pos_lst, visualization );
    if ( pos_lst.size() > 0 )
    {
        cout << "...[done] " << pos_lst.size() << " files." << endl;
    }
    else
    {
        cout << "no image in " << pos_dir <<endl;
        return 1;
    }
    Size pos_image_size = pos_lst[0].size();
    if ( detector_width && detector_height )
    {
        pos_image_size = Size( detector_width, detector_height );
    }
    else
    {
        for ( size_t i = 0; i < pos_lst.size(); ++i )
        {
            if( pos_lst[i].size() != pos_image_size )
            {
                cout << "All positive images should be same size!" << endl;
                exit( 1 );
            }
        }
        pos_image_size = pos_image_size / 8 * 8;
    }
    cout << "Negative images are being loaded...";
    load_images( neg_dir, full_neg_lst, visualization );
    cout << "...[done] " << full_neg_lst.size() << " files." << endl;
    cout << "Negative images are being processed...";
    sample_neg( full_neg_lst, neg_lst, pos_image_size );
    cout << "...[done] " << neg_lst.size() << " files." << endl;
    cout << "Histogram of Gradients are being calculated for positive images...";
    computeHOGs( pos_image_size, pos_lst, gradient_lst, flip_samples );
    size_t positive_count = gradient_lst.size();
    labels.assign( positive_count, +1 );
    cout << "...[done] ( positive images count : " << positive_count << " )" << endl;
    cout << "Histogram of Gradients are being calculated for negative images...";
    computeHOGs( pos_image_size, neg_lst, gradient_lst, flip_samples );
    size_t negative_count = gradient_lst.size() - positive_count;
    labels.insert( labels.end(), negative_count, -1 );
    CV_Assert( positive_count < labels.size() );
    cout << "...[done] ( negative images count : " << negative_count << " )" << endl;
    Mat train_data;
    convert_to_ml( gradient_lst, train_data );
    cout << "Training SVM...";
    Ptr< SVM > svm = SVM::create();
    /* Default values to train SVM */
    svm->setCoef0( 0.0 );
    svm->setDegree( 3 );
    svm->setTermCriteria( TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3 ) );
    svm->setGamma( 0 );
    svm->setKernel( SVM::LINEAR );
    svm->setNu( 0.5 );
    svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
    svm->setC( 0.01 ); // From paper, soft classifier
    svm->setType( SVM::EPS_SVR ); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
    svm->train( train_data, ROW_SAMPLE, labels );
    cout << "...[done]" << endl;

    HOGDescriptor hog;
    hog.winSize = pos_image_size;
    hog.setSVMDetector( get_svm_detector( svm ) );
    hog.save( obj_det_filename );
    test_trained_detector( obj_det_filename, test_dir );
    return 0;
}