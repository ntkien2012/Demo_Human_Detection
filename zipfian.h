typedef struct imageType{
	cv::Mat rawImg;
	std::vector<std::vector<int>> grayImg;
} imageType;

typedef struct imagePara{
    cv::Mat rgbImg;
    std::vector<std::vector<int>> It;
    std::vector<std::vector<int>> Mt;
    std::vector<std::vector<int>> Vt;
    std::vector<std::vector<int>> Ot;
    std::vector<std::vector<int>> Et;
} imagePara;

typedef struct pixPos{
    int x;
    int y;
} pixPos;

imageType rgbToGray(const cv::Mat& img) {
    imageType result;
    result.rawImg = img;
    // Ensure the input image is in RGB format
    if (img.channels() != 3) {
        throw std::runtime_error("Input image must be an RGB image.");
    }
    // Initialize the gray Mat with the same size as the input image and type CV_8UC1
    std::vector<std::vector<int>> gray(img.rows, std::vector<int>(img.cols));
    // Split into channels
    std::vector<cv::Mat> channels; 
    cv::split(img, channels);

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            // Calculate the average of the RGB components
            gray[i][j] = (((int)channels[0].at<uchar>(i, j) +  (int)channels[1].at<uchar>(i, j) +  (int)channels[2].at<uchar>(i, j)) / 3); // Red
        }
    }
    result.grayImg = gray;
    return result;
}

void releaseImagePara(imagePara &img) {
    img.rgbImg.release();
    img.It.clear();
    img.It.shrink_to_fit();
    img.Mt.clear();
    img.Mt.shrink_to_fit(); 
    img.Vt.clear();
    img.Vt.shrink_to_fit();
    img.Ot.clear();
    img.Ot.shrink_to_fit();
    img.Et.clear();
    img.Et.shrink_to_fit();
}

bool areAdjacent(pixPos a, pixPos b){

  int xDiff = abs(a.x - b.x); 
  int yDiff = abs(a.y - b.y);

  return (xDiff <= 1 && yDiff == 0) || (xDiff == 0 && yDiff <= 1); 
}

void postProcessImage(cv::Mat& image) {
    // Step 1: Remove stand-alone pixels
    cv::Mat element1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    //cv::morphologyEx(image, image, cv::MORPH_OPEN, element1);

    // Step 2: Apply 3x3 morphological closing
    //cv::morphologyEx(image, image, cv::MORPH_CLOSE, element1);

    
    cv::morphologyEx(image, image, cv::MORPH_OPEN, element1);                                                                                             
    cv::morphologyEx(image, image, cv::MORPH_CLOSE, element1);
    cv::dilate(image, element2, 1);
}

// Function to calculate the Euclidean distance between two points
double euclideanDistance(const cv::Point& p1, const cv::Point& p2) {
    return sqrt(pow(double(p1.x - p2.x), 2) + pow(double(p1.y - p2.y), 2));
}

// Function to merge two contours
std::vector<cv::Point> mergeContours(const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
    std::vector<cv::Point> merged;
    merged.reserve(c1.size() + c2.size()); // preallocate memory
    merged.insert(merged.end(), c1.begin(), c1.end());
    merged.insert(merged.end(), c2.begin(), c2.end());
    return merged;
}

// Function to reduce contours based on proximity
std::vector<std::vector<cv::Point>> reduceContours(const std::vector<std::vector<cv::Point>>& inputContours, double proximityThreshold) {
    std::vector<cv::Rect> boundingBoxes(inputContours.size());
    std::vector<cv::Point> centroids(inputContours.size());
    for (size_t i = 0; i < inputContours.size(); ++i) {
        boundingBoxes[i] = cv::boundingRect(inputContours[i]);
        centroids[i] = (boundingBoxes[i].tl() + boundingBoxes[i].br()) / 2;
    }

    std::vector<bool> merged(inputContours.size(), false);
    std::vector<std::vector<cv::Point>> mergedContours;

    for (size_t i = 0; i < inputContours.size(); ++i) {
        if (merged[i]) continue;

        auto contour1 = inputContours[i];
        for (size_t j = i + 1; j < inputContours.size(); ++j) {
            if (merged[j]) continue;

            if (euclideanDistance(centroids[i], centroids[j]) < proximityThreshold) {
                contour1 = mergeContours(contour1, inputContours[j]);
                merged[j] = true;
            }
        }
        mergedContours.push_back(contour1);
    }

    return mergedContours;
}

std::vector<std::vector<cv::Point>> mergeNearbyContours(const std::vector<std::vector<cv::Point>>& contours, int maxDist) {

    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point>> mergedContours;
    
    for(const auto& c : contours) {
        boxes.push_back(cv::boundingRect(c));
        mergedContours.push_back({c}); // initialize with original contour
    }
    
    for(size_t i = 0; i < boxes.size(); i++) {
        cv::Rect box = boxes[i];
        for(size_t j = i+1; j < boxes.size(); j++) {
            
            cv::Rect otherBox = boxes[j];
            cv::Point centeri(box.x + box.width/2, box.y + box.height/2);
            cv::Point centerj(otherBox.x + otherBox.width/2, otherBox.y + otherBox.height/2);

            double centerPointDist = std::sqrt(std::pow(centeri.x - centerj.x, 2) + std::pow(centeri.y - centerj.y, 2));
                        
            if(centerPointDist < maxDist) {               
                std::vector<cv::Point> newContour;
                newContour.insert(newContour.end(), mergedContours[i].begin(), mergedContours[i].end());
                newContour.insert(newContour.end(), mergedContours[j].begin(), mergedContours[j].end());
                mergedContours[i] = newContour;
                
                mergedContours.erase(mergedContours.begin()+j);
            }
        }
    }
    
    return mergedContours;
}

std::vector<std::vector<cv::Point>> eraseInnerRects(std::vector<std::vector<cv::Point>>& rectangles) {

  for(size_t i = 0; i < rectangles.size(); ++i) {
    for(size_t j = 0; j < rectangles.size(); ++j) {
      if(i == j) continue;

      auto& outer = rectangles[j];    
      auto& inner = rectangles[i];

      // Assuming the first point is top-left and the second is bottom-right
      bool isInner = 
        inner[0].x >= outer[0].x &&
        inner[0].y >= outer[0].y &&
        inner[1].x <= outer[1].x &&
        inner[1].y <= outer[1].y;

      if(isInner) {
        rectangles.erase(rectangles.begin() + i);
        i--; // Adjust the index after erasing
        break;
      }
    }
  }

  return rectangles;  
}


std::vector<std::vector<cv::Point>> processImage(imagePara& im0, imagePara& im1, int m, int TV, int N, int t, int Vmax, int Vmin) {
    // Step #0: Variance Threshold Computation
    int modValue = t % (int)std::pow(2, m);

    // Find the greatest 2^p that divides modValue
    // This is done by finding the rightmost set bit in modValue
    int p = 0;
    while (modValue > 0) {
        // If the least significant bit is set, we found our p
        if (modValue & 1) {
            break;
        }
        // Right shift modValue by 1 and increase p
        modValue >>= 1;
        p++;
    }

    int sigma = std::pow(2, m) / std::pow(2, p);

    // Step #1: Conditional Mt Estimation
    for (int x = 0; x < im1.It.size(); x++) {
        for (int y = 0; y < im1.It[0].size(); y++) {
            if (im0.Vt[x][y] > sigma) {
                if (im0.Mt[x][y] < im1.It[x][y]) {
                    im1.Mt[x][y] = im0.Mt[x][y] + 1;
                } else if (im0.Mt[x][y] > im1.It[x][y]) {
                    im1.Mt[x][y] = im0.Mt[x][y] - 1;
                } else {
                    im1.Mt[x][y] = im0.Mt[x][y];
                }
            } else {
                im1.Mt[x][y] = im0.Mt[x][y];
            }

            // Step #2: Ot Computation
            im1.Ot[x][y] = std::abs(im1.Mt[x][y] - im1.It[x][y]);

            // Step #3: Update Vt every TV frames
            if (t % TV == 0) {
                if (im0.Vt[x][y] < N * im1.Ot[x][y]) {
                    im1.Vt[x][y] = im0.Vt[x][y] + 1;
                } else if (im0.Vt[x][y] > N * im1.Ot[x][y]) {
                    im1.Vt[x][y] = im0.Vt[x][y] - 1;
                } else {
                    im1.Vt[x][y] = im0.Vt[x][y];
                }
                im1.Vt[x][y] = std::max(std::min(im1.Vt[x][y], Vmax), Vmin);
            }

            // Step #4: Eˆt Estimation
            im1.Et[x][y] = (im1.Ot[x][y] < im1.Vt[x][y]) ? 0 : 1;
        }
    }

    std::vector<cv::Rect> boundingBoxes;
    
    cv::Mat output(im1.Et.size(), im1.Et[0].size(), CV_8UC1, cv::Scalar(0)); // Blue color

    for (int i = 0; i < im1.Et.size(); ++i) {
        for (int j = 0; j < im1.Et[0].size(); ++j) {
            if (im1.Et[i][j] == 1) {
                output.at<uchar>(i,j) = 255; 
            } 
        }
    }

    //Extracting Moving Blocks
    // const int constThreshold = 2;
    // std::vector<std::vector<int>> moving(im1.It.size(), std::vector<int>(im1.It[0].size(), 0));
    // std::vector<std::vector<int>> sum(im1.It.size(), std::vector<int>(im1.It[0].size(), 0));
    // // std::vector<pixPos> movingList(im1.It.size() * im1.It[0].size());
    // // int numberOfMoving = 0;
    // for (int x = 1; x < im1.It.size() - 1; ++x) {
    //     for (int y = 1; y < im1.It[x].size() - 1; ++y) {
    //         //Check each neighbor of block x
    //         for (int dx = -1; dx <= 1; ++dx) {
    //             for (int dy = -1; dy <= 1; ++dy) {
    //                 int nx = x + dx;
    //                 int ny = y + dy;
    //                 if (nx >= 0 && nx < im1.It.size() && ny >= 0 && ny < im1.It[0].size()) {
    //                     // Counting temporary moving neighbors
    //                     if (im1.Et[nx][ny] == 1) { 
    //                         sum[nx][ny]++;
    //                     }
    //                 }
    //             }
    //         }
    //         // Determining if the block is moving or stationary
    //         if((sum[x][y] > constThreshold) || (im1.Et[x][y] == 1 && sum[x][y] > 0))
    //         //if((sum[x][y] > 8))
    //         {
    //           moving[x][y] = 1;
    //         }
    //         else moving[x][y] = 0;
    //     }
    // }

    // std::vector<cv::Rect> boundingBoxes;
    
    // cv::Mat output(moving.size(), moving[0].size(), CV_8UC1, cv::Scalar(0)); // Blue color

    // for (int i = 0; i < moving.size(); ++i) {
    //     for (int j = 0; j < moving[0].size(); ++j) {
    //         if (moving[i][j] == 1) {
    //             output.at<uchar>(i,j) = 255; 
    //         } 
    //     }
    // }
    
    postProcessImage(output);

    std::vector<std::vector<cv::Point>> contours;

    // Find contours and hierarchy
    cv::findContours(output, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // cv::imshow("Output2", output);
    // cv::waitKey(1);
    //auto reducedContours = mergeNearbyContours(contours, 10);

    // for(int i = contours.size()-1; i >= 0; i--) {
    //     cv::Rect rect = boundingRect(contours[i]);
    //     if(rect.width < 64 || rect.height < 128) { 
    //         contours.erase(contours.begin() + i);
    //     }
    // }

    // Reduce contours
    //auto reducedContours = reduceContours(contours, 3);
    //reducedContours = eraseInnerRects(reducedContours);
    return contours;
}

cv::Mat resizeImage(const cv::Mat& img, int newWidth, int newHeight) {
    // Create output image
    cv::Mat resized;
    resized.create(newHeight, newWidth, img.type());

    // Resize image using bilinear interpolation
    cv::resize(img, resized, resized.size(), 0, 0, cv::INTER_LINEAR); // Bilinear interpolation

    return resized;
}