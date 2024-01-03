#ifndef OVERLAP_H
#define OVERLAP_H

#include "csv.h"

std::vector<int> checkRectOverlap(const std::vector<double>& inRect, const std::vector<std::vector<double>>& compRects) {
  std::vector<int> indeces;
  // Get the coordinates of the top-left and bottom-right corners
  // of the input rectangle
  double a1x = inRect[0]; 
  double a1y = inRect[1];
  double a2x = a1x + inRect[2];
  double a2y = a1y + inRect[3];
  //std::cout << a1x << "\t" << a1y << "\t" << a2x << "\t" << a2y << std::endl;
  // Compute the area of the input rectangle
  double aArea = inRect[2] * inRect[3];
  // For each of the comparison rectangles
  for (size_t i = 0; i < compRects.size(); i++) {
    // Get the coordinates of the top-left and bottom-right corners
    double b1x = compRects[i][0];
    double b1y = compRects[i][1]; 
    double b2x = b1x + compRects[i][2];
    double b2y = b1y + compRects[i][3];
    //std::cout << b1x << "\t" << b1y << "\t" << b2x << "\t" << b2y << std::endl;
    // Compute the area of the comparison rectangle
    double bArea = compRects[i][2] * compRects[i][3];

    // Calculate overlap in x and y
    double xOverlap = std::max(0.0, std::min(a2x, b2x) - std::max(a1x, b1x));
    double yOverlap = std::max(0.0, std::min(a2y, b2y) - std::max(a1y, b1y));

    // Compute intersection area
    double intersectArea = xOverlap * yOverlap;

    // Compute union area 
    double unionArea = aArea + bArea - intersectArea;
    //std::cout << intersectArea / unionArea << std::endl;
    // Check if overlap exceeds threshold
    if (intersectArea / unionArea > 0.2) {
      indeces.push_back(i);
      //std::cout << i << std::endl;
    }
  }
  
  return indeces;
}


double IoU(const std::vector<double>& rectA, const std::vector<double>& rectB) {
    double xA = std::max(rectA[0], rectB[0]);
    double yA = std::max(rectA[1], rectB[1]);
    double xB = std::min(rectA[0] + rectA[2], rectB[0] + rectB[2]);
    double yB = std::min(rectA[1] + rectA[3], rectB[1] + rectB[3]);

    double interArea = std::max(0.0, xB - xA) * std::max(0.0, yB - yA);
    double boxAArea = rectA[2] * rectA[3];
    double boxBArea = rectB[2] * rectB[3];

    double iou = interArea / (boxAArea + boxBArea - interArea);
    return iou;
}

std::vector<std::vector<double>> nonMaxSuppression(std::vector<std::vector<double>>& rects, double overlapThreshold) {
    std::vector<std::vector<double>> output;

    while (!rects.empty()) {
        auto maxIt = std::max_element(rects.begin(), rects.end(), 
            [](const std::vector<double>& a, const std::vector<double>& b) {
                return a[2] * a[3] < b[2] * b[3]; // Assuming the score is the area
            });

        std::vector<double> maxRect = *maxIt;
        rects.erase(maxIt);

        output.push_back(maxRect);

        rects.erase(std::remove_if(rects.begin(), rects.end(), [&maxRect, overlapThreshold](const std::vector<double>& rect) {
                return IoU(maxRect, rect) > overlapThreshold;
            }), rects.end());
    }

    return output;
}
#endif // OVERLAP_H