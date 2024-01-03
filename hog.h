#ifndef HOG_H
#define HOG_H
#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h> 
#include <thread>
#include <mutex>

// Example HOG parameters
const int numVertCells = 16;
const int numHorizCells = 8; 
const int cellSize = 8;
const int numBins = 9;

const int debug = 0;

double degreeToRadian(double degree) {
  const double PI = 3.14159265;
  double radian = degree / 180.0 * PI;
  return radian;
}

#define COS10_256 252
#define COS30_256 222
#define COS50_256 165
#define COS70_256 88
#define COS90_256 0
#define COS110_256 -88
#define COS130_256 -165
#define COS150_256 -222
#define COS170_256 -252

#define SIN10_256 44
#define SIN30_256 128
#define SIN50_256 196
#define SIN70_256 241
#define SIN90_256 256
#define SIN110_256 241
#define SIN130_256 196
#define SIN150_256 128
#define SIN170_256 44

// Helper function to compute histogram
std::vector<double> getHistogramVer2(const std::vector<double>& magnit_1, const std::vector<double>& magnit_2,const std::vector<double>& angle_1, const std::vector<double>& angle_2, int numBins) {
  double angle;
  std::vector<double> histogram(numBins, 0);
  for(int i=0; i<64; i++) {
    if ((angle_1[i] == 10.0) && (angle_2[i] == 30.0)) {
      if(histogram[0] < 255){
        histogram[0] += round(magnit_1[i]/4);
      }
      if(histogram[1] < 255){
        histogram[1] += round(magnit_2[i]/4);
      }
    } else if ((angle_1[i] == 30.0) && (angle_2[i] == 50.0)) {
      if(histogram[1] < 255){
        histogram[1] += round(magnit_1[i]/4);
      }
      if(histogram[2] < 255){
        histogram[2] += round(magnit_2[i]/4);
      }
    } else if ((angle_1[i] == 50.0) && (angle_2[i] == 70.0)) {
      if(histogram[2] < 255){
        histogram[2] += round(magnit_1[i]/4);
      }
      if(histogram[3] < 255){
        histogram[3] += round(magnit_2[i]/4);
      }
    } else if ((angle_1[i] == 70.0) && (angle_2[i] == 90.0)) {
      if(histogram[3] < 255){
        histogram[3] += round(magnit_1[i]/4);
      }
      if(histogram[4] < 255){
        histogram[4] += round(magnit_2[i]/4);
      }
    } else if ((angle_1[i] == 110.0) && (angle_2[i] == 90.0)) {
      if(histogram[4] < 255){
        histogram[4] += round(magnit_1[i]/4);
      }
      if(histogram[5] < 255){
        histogram[5] += round(magnit_2[i]/4);
      }
    } else if ((angle_1[i] == 130.0) && (angle_2[i] == 110.0)) {
      if(histogram[5] < 255){
        histogram[5] += round(magnit_1[i]/4);
      }
      if(histogram[6] < 255){
        histogram[6] += round(magnit_2[i]/4);
      }
    } else if ((angle_1[i] == 150.0) && (angle_2[i] == 130.0)) {
      if(histogram[6] < 255){
        histogram[6] += round(magnit_1[i]/4);
      }
      if(histogram[7] < 255){
        histogram[7] += round(magnit_2[i]/4);
      }
    } else if ((angle_1[i] == 170.0) && (angle_2[i] == 150.0)) {
      if(histogram[7] < 255){
        histogram[7] += round(magnit_1[i]/4);
      }
      if(histogram[8] < 255){
        histogram[8] += round(magnit_2[i]/4);
      }
    } else{
      if(histogram[8] < 255){
        histogram[8] += round(magnit_1[i]/4);
      }
      if(histogram[0] < 255){
        histogram[0] += round(magnit_2[i]/4);
      }
    } 
  }
  return histogram;
}

std::vector<std::vector<std::vector<double>>> getHistogramsForImage(const std::vector<std::vector<double>>& img, int& xoffset, int& yoffset) {
  int imgHeight = img.size();
  int imgWidth = img[0].size();

  // Compute number of cells
  int numHorizCells = std::floor((imgWidth - 2) / cellSize);
  int numVertCells = std::floor((imgHeight - 2) / cellSize);

  // Compute new cropped image dimensions
  int newWidth = (numHorizCells * cellSize) + 2;
  int newHeight = (numVertCells * cellSize) + 2;

  // Compute crop offsets
  int xoffset_1, yoffset_1;
  xoffset_1 = std::round((imgWidth - newWidth) / 2.0);
  yoffset_1 = std::round((imgHeight - newHeight) / 2.0);
  xoffset = xoffset_1;
  yoffset = xoffset_1;

  // Assuming img, newHeight, newWidth, xoffset_1, and yoffset_1 are defined and valid
  std::vector<std::vector<double>> dx(newHeight - 2, std::vector<double>(newWidth - 2));
  std::vector<std::vector<double>> dy(newHeight - 2, std::vector<double>(newWidth - 2));

  for (int i = 1; i < newHeight - 1; i++) {
    for (int j = 1; j < newWidth - 1; j++) {
      dx[i - 1][j - 1] = img[i + yoffset_1][j + 1 + xoffset_1] - img[i + yoffset_1][j - 1 + xoffset_1];
      dy[i - 1][j - 1] = img[i + 1 + yoffset_1][j + xoffset_1] - img[i - 1 + yoffset_1][j + xoffset_1];
    }
  }
  //Compute angles
  
  std::vector<std::vector<double>> angle_1(dx.size(), std::vector<double>(dx[0].size()));
  std::vector<std::vector<double>> angle_2(dx.size(), std::vector<double>(dx[0].size()));
  
  // Compute magnitudes
  std::vector<std::vector<double>> magnit_1(dx.size(), std::vector<double>(dx[0].size(), 0));
  std::vector<std::vector<double>> magnit_2(dx.size(), std::vector<double>(dx[0].size(), 0));
  
  for(int i=0; i<dx.size(); i++) {
    for(int j=0; j<dx[0].size(); j++) {
      double X = dx[i][j];
      double Y = dy[i][j];
      
      if (17*abs(Y) >= 3*abs(X)) {
        if(Y * X < 0){
          magnit_1[i][j] = (X * SIN150_256 - Y * COS150_256)/256;
          magnit_2[i][j] = (Y * COS170_256 - X * SIN170_256)/256;
          angle_1[i][j] = 170;
          angle_2[i][j] = 150;
        }else{
          magnit_1[i][j] = (X * SIN30_256 - Y * COS30_256)/256;
          magnit_2[i][j] = (Y * COS10_256 - X * SIN10_256)/256;
          angle_1[i][j] = 10;
          angle_2[i][j] = 30;
        }
      } else if (168*abs(Y) >= 97*abs(X)) {
        if(Y * X < 0){
          magnit_1[i][j] = (X * SIN130_256 - Y * COS130_256)/256;
          magnit_2[i][j] = (Y * COS150_256 - X * SIN150_256)/256;
          angle_1[i][j] = 150;
          angle_2[i][j] = 130;
        }else{
          magnit_1[i][j] = (X * SIN50_256 - Y * COS50_256)/256;
          magnit_2[i][j] = (Y * COS30_256 - X * SIN30_256)/256;
          angle_1[i][j] = 30;
          angle_2[i][j] = 50;
        }
      } else if (73*abs(Y) >= 87*abs(X)) {
        if(Y * X < 0){
          magnit_1[i][j] = (X * SIN110_256 - Y * COS110_256)/256;
          magnit_2[i][j] = (Y * COS130_256 - X * SIN130_256)/256;
          angle_1[i][j] = 130;
          angle_2[i][j] = 110;
        }else{
          magnit_1[i][j] = (X * SIN70_256 - Y * COS70_256)/256;
          magnit_2[i][j] = (Y * COS50_256 - X * SIN50_256)/256;
          angle_1[i][j] = 50;
          angle_2[i][j] = 70;
        }
      } else if (4 * abs(Y) >= 11*abs(X)) {
        if(Y * X < 0){
          magnit_1[i][j] = (X * SIN90_256 - Y * COS90_256)/256;
          magnit_2[i][j] = (Y * COS110_256 - X * SIN110_256)/256;
          angle_1[i][j] = 110;
          angle_2[i][j] = 90;
        }else{
          magnit_1[i][j] = (X * SIN90_256 - Y * COS90_256)/256;
          magnit_2[i][j] = (Y * COS70_256 - X * SIN70_256)/256;
          angle_1[i][j] = 70;
          angle_2[i][j] = 90;
        }
      } else{
        angle_1[i][j] = 170.0;
        angle_2[i][j] = 10.0;
      }

      if (magnit_1[i][j] < 0) magnit_1[i][j] = 0;
      if (magnit_2[i][j] < 0) magnit_2[i][j] = 0;
    }
  }

  // Compute histograms
  std::vector<std::vector<std::vector<double>>> histograms(numVertCells, std::vector<std::vector<double>>(numHorizCells,std::vector<double>(numBins, 0)));
  std::vector<double> cellAngle_1(cellSize * cellSize);
  std::vector<double> cellAngle_2(cellSize * cellSize);
  std::vector<double> cellMagnitudes_1(cellSize * cellSize);
  std::vector<double> cellMagnitudes_2(cellSize * cellSize);
                                                                                         
   for (int row = 0; row < numVertCells; row++) {
    int rowOffset = (row * cellSize);
    for (int col = 0; col < numHorizCells; col++) {
      int colOffset = (col * cellSize);
      int count = 0; 
      for(int rowIndeces = 0; rowIndeces < cellSize; rowIndeces++) {
        for(int colIndeces = 0; colIndeces < cellSize; colIndeces++) {
          cellAngle_1[count] = angle_1[rowIndeces + rowOffset][colIndeces + colOffset];
          cellAngle_2[count] = angle_2[rowIndeces + rowOffset][colIndeces + colOffset];
          cellMagnitudes_1[count] = magnit_1[rowIndeces + rowOffset][colIndeces + colOffset];
          cellMagnitudes_2[count] = magnit_2[rowIndeces + rowOffset][colIndeces + colOffset];
          count++;
        }
      }
      // Compute histogram
      histograms[row][col] = getHistogramVer2(cellMagnitudes_1, cellMagnitudes_2, cellAngle_1, cellAngle_2, numBins);
    }
  }
  return histograms;
}

std::vector<double> getDescriptorFromHistograms(const std::vector<std::vector<std::vector<double>>>& histograms) {
  std::vector<double> descriptor;
  std::vector<double> blockHist(36);
  for(int row = 0; row < numVertCells-1; row++) {
    for(int col = 0; col < numHorizCells-1; col++) {
      for(int i = 0; i < 9; i++) {
        if (col % 2 == 0) {
          blockHist[i] = histograms[row][col][i];
          blockHist[9+i] = histograms[row][col+1][i];  
          blockHist[18+i] = histograms[row+1][col][i];
          blockHist[27+i] = histograms[row+1][col+1][i];
        } else {
          blockHist[i] = histograms[row][col+1][i];
          blockHist[9+i] = histograms[row][col][i];
          blockHist[18+i] = histograms[row+1][col+1][i];
          blockHist[27+i] = histograms[row+1][col][i];        
        }
      }
      double l2_norm_sqrt = cv::norm(blockHist) + 0.01;      
      std::vector<double> l2_normalized(36);
      for(int i=0; i<36; i++) {
        l2_normalized[i] = blockHist[i] / l2_norm_sqrt; 
      }

      descriptor.insert(descriptor.end(), l2_normalized.begin(), l2_normalized.end());
    }
  }
  return descriptor;
}

// Get descriptor for a sub-region 
std::vector<double> getDescriptorForRegion(const std::vector<std::vector<std::vector<double>>>& allHistograms,int rowStart, int colStart) {
  std::vector<std::vector<std::vector<double>>> histograms(numVertCells, std::vector<std::vector<double>>(numHorizCells, std::vector<double>(numBins, 0)));
  std::vector<double> description;
  for(int i = 0; i < numVertCells; i++) {
    for(int j = 0; j < numHorizCells; j++) {
      for(int k = 0; k < numBins; k++){
        histograms[i][j][k] =  allHistograms[rowStart+i][colStart+j][k];
      }
    }
  }
  description = getDescriptorFromHistograms(histograms);
  return  description;
}

std::vector<double> exportThetaData() {
  std::vector<double> theta1;
  std::string line;
  
  // Load theta data from hog_model.txt into vector
  std::ifstream fin("hog_model.txt");
  if (!fin) {
    std::cerr << "Error opening file." << std::endl;
  }
  
  while (getline(fin, line)) {
    // process line
    std::stringstream linestream(line);
    double value;
    while (linestream >> value) {
      theta1.push_back(value);
    }
  }
  fin.close();
  return theta1;
}

// Function to multiply a transposed single-row matrix (vector) by another vector
double multiply_transposed_vector(const std::vector<double>& row_vector, const std::vector<double>& vector) {
    if (row_vector.size() != vector.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }

    double result = 0.0;
    for (size_t i = 0; i < row_vector.size(); ++i) {
        result += row_vector[i] * vector[i];
    }
    return result;
}

double computeP(const std::vector<double>& H, const std::vector<double>& theta) {
  // Validate input dimensions
  int n = H.size();
  if (theta.size() != n) {
    throw "Error: H and theta must have the same length";
  }
  // Compute dot product
  double p = 0;
  for (int i = 0; i < n; i++) {
    p += H[i] * theta[i]; 
  }
  return p; 
}

cv::Mat resizeImage(const cv::Mat& img, double scaleFactor) {
    // Get image size
    int origWidth = img.cols;
    int origHeight = img.rows;

    // Compute new size
    int newWidth = cvRound(origWidth * scaleFactor);
    int newHeight = cvRound(origHeight * scaleFactor);

    // Create output image
    cv::Mat resized;
    resized.create(newHeight, newWidth, img.type());

    // Resize image using bilinear interpolation
    cv::resize(img, resized, resized.size(), 0, 0, cv::INTER_LINEAR); // Bilinear interpolation

    return resized;
}

std::vector<std::vector<double>> rgb2gray(const cv::Mat& img) {
    // Ensure the input image is in RGB format
    if (img.channels() != 3) {
        throw std::runtime_error("Input image must be an RGB image.");
    }
    // Initialize the gray Mat with the same size as the input image and type CV_8UC1
    std::vector<std::vector<double>> gray(img.rows, std::vector<double>(img.cols));
    // Split into channels
    std::vector<cv::Mat> channels; 
    cv::split(img, channels);

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            // Calculate the average of the RGB components
            gray[i][j] = (((double)channels[0].at<uchar>(i, j) +  (double)channels[1].at<uchar>(i, j) +  (double)channels[2].at<uchar>(i, j)) / 3.0); // Red
        }
    }
    return gray;
}

std::mutex regions_mutex; // Mutex for thread-safe access to 'regions'

void processScale(cv::Mat& img, double scale, const std::vector<double>& theta, int numVertCells, int numHorizCells, int cellSize, double threshold, std::vector<std::vector<double>>& regions, std::mutex& regions_mutex) {
  int xoffset = 0, yoffset = 0;
  int xstart, ystart;
  
  cv::Mat img_resized = resizeImage(img, scale); // Assuming resizeImage is defined elsewhere
  if (img_resized.cols < 70 || img_resized.rows < 135) {
      return;
  }

  std::vector<std::vector<double>> gray = rgb2gray(img_resized); // Assuming rgb2gray is defined elsewhere
  std::vector<std::vector<std::vector<double>>> allHistograms = getHistogramsForImage(gray, xoffset, yoffset);
  //std::cout << allHistograms.size() << "\t" << allHistograms.size()
  for (int i = 0; i < allHistograms.size() - numVertCells; i++) {
    for (int j = 0; j < allHistograms[0].size() - numHorizCells; j++) {
      std::vector<double> histograms_1 = getDescriptorForRegion(allHistograms, i, j); // Assuming getDescriptorForRegion is defined elsewhere
      double p = computeP(histograms_1, theta); // Assuming computeP is defined elsewhere
      if (p > threshold) {
        xstart = xoffset + (j * cellSize);
        ystart = yoffset + (i * cellSize);
        std::vector<double> rect = {round(xstart / scale), round(ystart / scale), round((numHorizCells * cellSize + 2) / scale), round((numVertCells * cellSize + 2) / scale), p};
        
        std::lock_guard<std::mutex> lock(regions_mutex); // Thread-safe access to 'regions'
        regions.push_back(rect);
      }
    }
  }
}

std::vector<std::vector<double>> searchImage(cv::Mat& img, std::vector<double>& theta) {
  int numberScale = 32;
  double threshold = 0.4;
  int numVertCells = 16; // Number of vertical cells
  int numHorizCells = 8; // Number of horizontal cells
  int cellSize = 8; // Size of each cell
  std::vector<std::vector<double>> regions;
  std::mutex regions_mutex;

  const int numThreads = 6; // Number of threads
  std::vector<std::thread> threads;

  for (int k = 0; k < numThreads; k++) {
    threads.emplace_back([&img, &theta, &regions, &regions_mutex, numVertCells, numHorizCells, cellSize, threshold, k, numberScale]() {
      // std::cout << "Thread " << k << " started" << std::endl;
      for (int i = k; i < numberScale; i += numThreads) {
          // std::cout << "Thread " << k << " processing scale " << i << std::endl;
          double scale = std::pow(1.05, -i);
          processScale(img, scale, theta, numVertCells, numHorizCells, cellSize, threshold, regions, regions_mutex);
      }
    });
  }

  // Join all threads
  for (auto& thread : threads) {
    if (thread.joinable()) {
        thread.join();
    }
  }
  return regions;
}
#endif // HOG_H