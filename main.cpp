#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <numeric>
#include <thread>
#include <mutex>
#include "csv.h"
#include "zipfian.h"
#include "hogsvm.h"
#include "overlap.h"

namespace fs = std::filesystem;

int main() {
    int m = 8, TV = 1, N = 2;
#if 1
    std::string path = "MOT15/train/PETS09-S2L1/img1"; // Replace with your images directory

    // Create a vector to hold the file names
    std::vector<std::string> fileNames;
    // Load file names into the vector
    for (const auto& entry : fs::directory_iterator(path)) {
        fileNames.push_back(entry.path().string());
    }

    // Sort the file names
    std::sort(fileNames.begin(), fileNames.end());
    cv::Mat image, nextImage;
    imageType grayImage_1, grayImage_2;
    
    std::vector<std::vector<cv::Point>> contourImage;
    // Load and convert images to grayscale
    std::vector<std::vector<double>> rectResult;
    std::vector<std::vector<double>> rectResultOverLap;
    std::vector<cv::Mat> croppedImages;
    imagePara temp;
    std::vector<double> theta;
    theta = exportThetaData();

    std::mutex mutexFrame;
    std::vector<std::thread> threadsFrame;
    for (int p = 1; p < (int)fileNames.size() - 1; p++) {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<imagePara> imageP(2);
        // if (p%2 == 0){
        //     continue;
        // }
        double scale = 1.5;
        image = resizeImage(cv::imread(fileNames[p - 1], cv::IMREAD_COLOR), cv::imread(fileNames[p - 1], cv::IMREAD_COLOR).cols * scale, cv::imread(fileNames[p - 1], cv::IMREAD_COLOR).rows * scale);
        nextImage = resizeImage(cv::imread(fileNames[p], cv::IMREAD_COLOR), cv::imread(fileNames[p], cv::IMREAD_COLOR).cols * scale, cv::imread(fileNames[p], cv::IMREAD_COLOR).rows * scale);
        grayImage_1 = rgbToGray(image);
        grayImage_2 = rgbToGray(nextImage);
        imageP[0].rgbImg = grayImage_1.rawImg;
        imageP[0].It = grayImage_1.grayImg;
        
        if(p == 1){
            imageP[0].Mt = imageP[0].It;  // Duplicate the image for other members if needed
            imageP[0].Vt = std::vector<std::vector<int>>(imageP[0].It.size(), std::vector<int>(imageP[0].It[0].size(), 2));
            imageP[0].Ot =  std::vector<std::vector<int>>(imageP[0].It.size(), std::vector<int>(imageP[0].It[0].size(), 0));
            imageP[0].Et =  std::vector<std::vector<int>>(imageP[0].It.size(), std::vector<int>(imageP[0].It[0].size(), 0));
        }else{
            imageP[0].Mt = temp.Mt;
            imageP[0].Vt = temp.Vt;
            imageP[0].Ot = temp.Ot;
            imageP[0].Et = temp.Et;
        }
        imageP[1].rgbImg = grayImage_2.rawImg;
        imageP[1].It = grayImage_2.grayImg;
        imageP[1].Mt = std::vector<std::vector<int>>(imageP[1].It.size(), std::vector<int>(imageP[1].It[0].size()));
        imageP[1].Vt = std::vector<std::vector<int>>(imageP[1].It.size(), std::vector<int>(imageP[1].It[0].size()));
        imageP[1].Ot = std::vector<std::vector<int>>(imageP[1].It.size(), std::vector<int>(imageP[1].It[0].size()));
        imageP[1].Et = std::vector<std::vector<int>>(imageP[1].It.size(), std::vector<int>(imageP[1].It[0].size()));
        //auto start_1 = std::chrono::high_resolution_clock::now();
        contourImage = processImage(imageP[0], imageP[1], m, TV, N, p + 1, 255, 2);
        temp = imageP[1];
        // auto end_1 = std::chrono::high_resolution_clock::now();
        // auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_1 - start_1); 
        // std::cout << "Zipfian: " << duration_1.count() << std::endl;

        for(size_t j = 0; j < contourImage.size(); j++)  
        {
            cv::Scalar color(255,0,0); // blue color       
            cv::Rect rectMotion = boundingRect(contourImage[j]);    
            //rectangle(imageP[1].rgbImg, rectMotion, color, 2);
            cv::Mat cropped = imageP[1].rgbImg(rectMotion);
            rectResult = searchImage(cropped, theta);
            rectResultOverLap = nonMaxSuppression(rectResult, 0.2);
            // for(int k= 0; k < rectResult.size(); k++){
            //   cv::Rect rect((int)rectResult[k][0] + rectMotion.x, (int)rectResult[k][1] + rectMotion.y, (int)rectResult[k][2], (int)rectResult[k][3]);
            //   cv::rectangle(imageP[1].rgbImg, rect, cv::Scalar(255,255,120), 1);
            // }
            for(int k= 0; k < rectResultOverLap.size(); k++){
              cv::Rect rect((int)rectResultOverLap[k][0] + rectMotion.x, (int)rectResultOverLap[k][1] + rectMotion.y, (int)rectResultOverLap[k][2], (int)rectResultOverLap[k][3]);
              cv::rectangle(imageP[1].rgbImg, rect, cv::Scalar(255,255,120), 1);
              std::cout << p << "," << -1 << "," << (double)((int)rectResultOverLap[k][0] + rectMotion.x)/scale << "," << (double)((int)rectResultOverLap[k][1] + rectMotion.y)/scale << "," << (double)((int)rectResultOverLap[k][2])/scale << "," <<  (double)((int)rectResultOverLap[k][3])/scale << std::endl;
            }
        }
        cv::imshow("Output1", imageP[1].rgbImg);
        cv::waitKey(1);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); 
        //std::cout << duration.count() <<"," << rectResult.size() << std::endl;
    }
    return 0;
#endif
}
