#ifndef CSV_H
#define CSV_H

#include <fstream>
#include <string>
#include <vector>

void write3DVectorToCSV(const std::vector<std::vector<std::vector<double>>>& data, std::string filename);
void write2DVectorToCSV(const std::vector<std::vector<double>>& data,  std::string filename);
void write1DVectorToCSV(const std::vector<double>& data, std::string filename);
std::vector<std::vector<double>> loadCSV(const std::string& filename);

void write3DVectorToCSV(const std::vector<std::vector<std::vector<double>>>& data, std::string filename) {
  std::ofstream file(filename);

  for (size_t i = 0; i < data.size(); i++) {
    for (size_t j = 0; j < data[i].size(); j++) {
      for (size_t k = 0; k < data[i][j].size(); k++) {
        file << data[i][j][k];
          file << ","; 
      }
    }
    file << "\n";
  }
  file.close();
}

void write2DVectorToCSV(const std::vector<std::vector<double>>& data, std::string filename) {
  std::ofstream file(filename);
  for (size_t i = 0; i < data.size(); i++) {
    for (size_t j = 0; j < data[i].size(); j++) {
      file << data[i][j];
      if (j < data[i].size() - 1) {
        file << ","; 
      }
    }
    file << "\n";
  }
  file.close();
}

void write2DVectorToCSV_int(const std::vector<std::vector<int>>& data, std::string filename) {
  std::ofstream file(filename);
  for (size_t i = 0; i < data.size(); i++) {
    for (size_t j = 0; j < data[i].size(); j++) {
      file << data[i][j];
      if (j < data[i].size() - 1) {
        file << ","; 
      }
    }
    file << "\n";
  }
  file.close();
}

void write1DVectorToCSV(const std::vector<double>& data, std::string filename) {
  std::ofstream file(filename);
  for (size_t i = 0; i < data.size(); i++) {
      file << data[i];
      file << ",";
    }
  file.close();
}

std::vector<std::vector<double>> loadCSV(const std::string& filename) {
  std::ifstream file(filename);
  std::vector<std::vector<double>> data;
  if (file) {
    std::string line;
    while (std::getline(file, line)) {
      std::stringstream linestream(line);
      std::vector<double> row;
      std::string field;
      while(std::getline(linestream, field, ',')) {
        row.push_back(std::stod(field));
      }
      data.push_back(row);
    }
  }
  file.close();
  return data;
}

#endif // CSV_H