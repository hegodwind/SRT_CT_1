#include <opencv2/opencv.hpp>

#include <filesystem>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm> 
#include <cstring>
#include <complex>
#include <string>
#include "nlohmann/json.hpp"



using namespace cv;
using namespace std;
using namespace dlib;
using json = nlohmann::json;

typedef dlib::matrix<double, 1, 1> input_vector;
typedef dlib::matrix<double, 4, 1> parameter_vector;

// 强制编译器以1字节对齐，确保内存布局与文件格式完全一致
#pragma pack(push, 1)

struct CsrHeader {
    char      tag;           // 0x00: 文件标签
    uint16_t  version;          // 0x04: 主版本号
    uint16_t  minVersion;       // 0x06: 副版本号
    uint32_t  fileLength;       // 0x08: 文件总长度
    uint32_t  headLength;       // 0x0C: 文件头长度
    uint32_t  extInfoLength;    // 0x10: 扩展信息长度
    uint32_t  markInfoLength;   // 0x14: 结论信息长度
    uint32_t  dataOffset;       // 0x18: 图像数据偏移量
    uint32_t  extInfoOffset;    // 0x1C: 扩展信息偏移量
    uint32_t  markInfoOffset;   // 0x20: 结论信息偏移量
    uint16_t  pixelType;        // 0x24: 像素类型
    uint16_t  pixelBits;        // 0x26: 每像素位数
    uint16_t  pixelBytes;       // 0x28: 每像素字节数
    uint16_t  pixelOrder;       // 0x2A: 像素顺序
    uint16_t  compression;      // 0x2C: 压缩标志
    uint16_t  reserve1;         // 0x2E: 保留字段
    double    maxPixel;         // 0x30: 最大像素值
    double    minPixel;         // 0x38: 最小像素值
    uint32_t  xLength;          // 0x40: 图像宽度
    uint32_t  yLength;          // 0x44: 图像高度
    uint32_t  zLength;          // 0x48: Z轴高度
    float     fPixelSizeX;      // 0x4C: X方向像素尺寸
    float     fPixelSizeY;      // 0x50: Y方向像素尺寸
    float     fPixelSizeZ;      // 0x54: Z方向像素尺寸
    uint32_t  nDataInfoLength;  // 0x58: 数据信息长度
    uint16_t  dataType;         // 0x5C: 数据类型
    uint16_t  CodeInfoLength;   // 0x5E: 代码信息长度
    uint32_t  filePad;       // 0x60: 保留字段
    float     startAngle;       // 0x78: 起始角度
    float     fStartRadius;     // 0x7C: 起始半径
};

#pragma pack(pop)

// --- 字节序处理辅助函数 (兼容版) ---

// 在程序运行时检测当前系统的字节序
bool is_host_little_endian() {
    uint16_t i = 1;
    // 如果整数1的第一个字节是1，则为小端序
    return *(reinterpret_cast<char*>(&i));
}

// 手动进行字节交换的模板函数
template <typename T>
T byteswap(T value) {
    // 对于未知类型，默认不进行任何操作
    return value;
}

// 针对uint16_t的特化版本
template <>
inline uint16_t byteswap<uint16_t>(uint16_t value) {
    return (value >> 8) | (value << 8);
}

// 针对uint32_t的特化版本
template <>
inline uint32_t byteswap<uint32_t>(uint32_t value) {
    value = ((value << 8) & 0xFF00FF00) | ((value >> 8) & 0x00FF00FF);
    return (value << 16) | (value >> 16);
}

// 针对uint64_t的特化版本
template <>
inline uint64_t byteswap<uint64_t>(uint64_t value) {
    value = ((value << 8) & 0xFF00FF00FF00FF00ULL) | ((value >> 8) & 0x00FF00FF00FF00FFULL);
    value = ((value << 16) & 0xFFFF0000FFFF0000ULL) | ((value >> 16) & 0x0000FFFF0000FFFFULL);
    return (value << 32) | (value >> 32);
}

// 针对float的特化版本
template <>
inline float byteswap<float>(float value) {
    uint32_t int_repr;
    memcpy(&int_repr, &value, sizeof(float));
    int_repr = byteswap<uint32_t>(int_repr);
    memcpy(&value, &int_repr, sizeof(float));
    return value;
}

// 针对double的特化版本
template <>
inline double byteswap<double>(double value) {
    uint64_t int_repr;
    memcpy(&int_repr, &value, sizeof(double));
    int_repr = byteswap<uint64_t>(int_repr);
    memcpy(&value, &int_repr, sizeof(double));
    return value;
}


// 假设文件字节序为小端序
const bool HOST_IS_LITTLE_ENDIAN = is_host_little_endian();

// 通用的、字节序感知的字段读取函数 (兼容版)
template <typename T>
T read_and_convert(std::ifstream& file) {
    T value;
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!file) {
        throw std::runtime_error("Error reading from file stream.");
    }

    // 如果主机是大端序，则需要进行字节交换
    if (!HOST_IS_LITTLE_ENDIAN) {
        return byteswap(value);
    }
    return value;
}

// 主加载函数：读取CSR文件并返回一个OpenCV Mat对象
cv::Mat loadCsrFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // 1. 解析文件头
    CsrHeader header;
    header.tag = read_and_convert<uint32_t>(file);
    header.version = read_and_convert<uint16_t>(file);
    header.minVersion = read_and_convert<uint16_t>(file);
    header.fileLength = read_and_convert<uint32_t>(file);
    header.headLength = read_and_convert<uint32_t>(file);
    header.extInfoLength = read_and_convert<uint32_t>(file);
    header.markInfoLength = read_and_convert<uint32_t>(file);
    header.dataOffset = read_and_convert<uint32_t>(file);
    header.extInfoOffset = read_and_convert<uint32_t>(file);
    header.markInfoOffset = read_and_convert<uint32_t>(file);
    header.pixelType = read_and_convert<uint16_t>(file);
    header.pixelBits = read_and_convert<uint16_t>(file);
    header.pixelBytes = read_and_convert<uint16_t>(file);
    header.pixelOrder = read_and_convert<uint16_t>(file);
    header.compression = read_and_convert<uint16_t>(file);
    header.reserve1 = read_and_convert<uint16_t>(file);
    header.maxPixel = read_and_convert<double>(file);
    header.minPixel = read_and_convert<double>(file);
    header.xLength = read_and_convert<uint32_t>(file);
    header.yLength = read_and_convert<uint32_t>(file);
    header.zLength = read_and_convert<uint32_t>(file);
    header.fPixelSizeX = read_and_convert<float>(file);
    header.fPixelSizeY = read_and_convert<float>(file);
    header.fPixelSizeZ = read_and_convert<float>(file);
    
    

    
    file.seekg(header.dataOffset, std::ios::beg);
    const size_t total_pixels = static_cast<size_t>(header.xLength) * header.yLength;
    std::vector<float> pixel_data(total_pixels);

    for (size_t i = 0; i < total_pixels; ++i) {
        pixel_data[i] = read_and_convert<float>(file);
    }

    
    cv::Mat mat_wrapper(header.yLength, header.xLength, CV_32FC1, pixel_data.data());
    std::vector<float> pixel_data_copy;
    for(int i=0;i< mat_wrapper.rows;i++){
        for(int j=0;j< mat_wrapper.cols;j++){
            pixel_data_copy.push_back(mat_wrapper.at<float>(i,j));
        }
    }

    // 使用.clone()进行深拷贝，生成一个内存自管理的Mat对象
    
    return mat_wrapper.clone();
}

// 进行窗口内的三次多项式最小二乘拟合，并返回窗口中点处的拟合值
double cubicFitAtCenter(const std::vector<double>& data, int startIdx, int windowSize) {
    
    Mat A(windowSize, 4, CV_64F);
    Mat y(windowSize, 1, CV_64F);
    
    for (int j = 0; j < windowSize; j++) {
        double x = j; 
        A.at<double>(j, 0) = 1.0;
        A.at<double>(j, 1) = x;
        A.at<double>(j, 2) = x * x;
        A.at<double>(j, 3) = x * x * x;
        y.at<double>(j, 0) = data[startIdx + j];
    }
    
    
    Mat coeff;
    solve(A, y, coeff, DECOMP_SVD);
    
    int center = windowSize / 2;
    double x_center = center;
    double fittedVal = coeff.at<double>(0,0)
                       + coeff.at<double>(1,0) * x_center
                       + coeff.at<double>(2,0) * x_center * x_center
                       + coeff.at<double>(3,0) * x_center * x_center * x_center;
    return fittedVal;
}

// 计算PSF：使用三次多项式拟合ERF，然后求导
std::vector<double> calculatePSF(const std::vector<double>& erf, int windowSize = 21) {
    std::vector<double> psf;
    int halfWin = windowSize / 2;
    
    
    for (int i = 0; i <= erf.size() - windowSize; i++) {
        
        Mat A(windowSize, 4, CV_64F);
        Mat y(windowSize, 1, CV_64F);
        
        for (int j = 0; j < windowSize; j++) {
            double x = j;
            A.at<double>(j, 0) = 1.0;
            A.at<double>(j, 1) = x;
            A.at<double>(j, 2) = x * x;
            A.at<double>(j, 3) = x * x * x;
            y.at<double>(j, 0) = erf[i + j];
        }
        
        
        Mat coeff;
        solve(A, y, coeff, DECOMP_SVD);
        
        
        double x_center = halfWin;
        double derivative = coeff.at<double>(1,0) 
                           + 2 * coeff.at<double>(2,0) * x_center 
                           + 3 * coeff.at<double>(3,0) * x_center * x_center;
                           
        psf.push_back(derivative);
    }
    
    return psf;
}



void my_fft(std::vector<std::complex<double>>& a) {
    int n = a.size();
    if (n <= 1) return;

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
		double Mi_PI = 3.14159265358979323846;   
        double ang = -2 * Mi_PI / len;
        std::complex<double> wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1);
            for (int j = 0; j < len / 2; j++) {
                std::complex<double> u = a[i + j];
                std::complex<double> v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// 计算MTF：使用自定义FFT计算PSF的离散傅里叶变换的幅度
std::vector<double> calculateMTF(const parameter_vector& param , const double& binsize) {
    double sigma = param(3)*binsize;
    int mtf_size = 100/sigma;
    std::vector<double> mtf(mtf_size);
    for (int i = 0; i < mtf_size; i++) {
        mtf[i] = exp(-2*pi*pi*sigma*sigma*i*i/10000);
    }
    return mtf;
}


//霍夫圆检测的类，其中包含原始数据
class CircleDetector {
public:  
    CircleDetector(const string& csrPath) {
        this->original_data  = loadCsrFile(csrPath);
        this->width = original_data.cols;
        this->height = original_data.rows;
        Mat temp = original_data.clone();
        temp.convertTo(this->Image, CV_8UC1, 255.0);
        temp.release();
    }
    Mat original_data;
    Mat Image;
    std::vector<Vec3f> circles;
    int width;
    int height;

    void detectCircles() {
        //模糊
        if (this->Image.empty()) {
            cout << "无法读取图像" << endl;
            return;
        }
        resize(this->Image, this->Image, Size(this->width / 4, this->height / 4));
        normalize(this->Image, this->Image, 0, 255, NORM_MINMAX);
        Mat blurImage;
        GaussianBlur(this->Image, blurImage, Size(9, 9), 0);

        int height = blurImage.rows;
        int weight = blurImage.cols;
        int minDist = 30;   // 最小距离
        int param1 = 100;     // Canny边缘检测高阈值  
        int param2 = 60;     // Hough变换高阈值
        int minRadius = 100;   // 最小半径
        int maxRadius = 3000;   // 最大半径

        HoughCircles(blurImage, this->circles, HOUGH_GRADIENT, 1, minDist, param1, param2, minRadius, maxRadius);

        cvtColor(this->Image, this->Image, COLOR_GRAY2BGR);
		
		//将检测出的圆画出来
        for (size_t i = 0; i < this->circles.size(); i++)
        {
            Point center(cvRound(this->circles[i][0]), cvRound(this->circles[i][1]));
            int radius = cvRound(this->circles[i][2]);
            circle(this->Image, center, radius, Scalar(0, 255, 0), 2, LINE_AA);
            circle(this->Image, center, 3, Scalar(0, 0, 255), -1, LINE_AA);
        }


        for (size_t i = 0; i < this->circles.size(); i++) {
            Vec3f& circle = this->circles[i];
            circle[0] *= 4;
            circle[1] *= 4;
            circle[2] *= 4;
        }
    }
};

class ERFCalculator {
private:
    float binSize;
    float innerRadius;
    float outerRadius;
    int numBins;
    float centerX;
    float centerY;

    std::vector<double> erf;
    std::vector<double> binSums;
    std::vector<int> binCounts;
    std::vector<double> erfSmoothed;

public:
    
    ERFCalculator(const CircleDetector& detector, float binSize = 0.05 ,int windowSize = 200) {
        
        this->binSize = binSize;
        this->innerRadius = detector.circles[0][2] - windowSize;
        this->outerRadius = detector.circles[0][2] + windowSize;
        this->numBins = static_cast<int>((outerRadius - innerRadius) / binSize) + 1;
        
        
        this->centerX = detector.circles[0][0];
        this->centerY = detector.circles[0][1];
        
       
        this->erf.resize(numBins, 0.0);
        this->binSums.resize(numBins, 0.0);
        this->binCounts.resize(numBins, 0);
    }
    
    // 计算初始ERF函数
    void calculateInitialERF(const Mat& image) {
        // 重置数据
        fill(binSums.begin(), binSums.end(), 0.0);
        fill(binCounts.begin(), binCounts.end(), 0);
        fill(erf.begin(), erf.end(), 0.0);


        // 遍历图像像素计算每个bin的像素平均值
          for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                double dx = x - centerX;
                double dy = y - centerY;
                double distance = sqrt(dx * dx + dy * dy);
                
                if(distance >= innerRadius && distance <= outerRadius) {
                    int binIndex = static_cast<int>((distance - innerRadius) / binSize);
                    if(binIndex >= 0 && binIndex < numBins) {
                        binSums[binIndex] += static_cast<float>(image.at<float>(y, x));
                        binCounts[binIndex]++;
                    }
                }
            }
        }
        
        // 计算每个bin的平均值
        for (int i = 0; i < numBins; i++) {
            if(binCounts[i] > 0)
                erf[i] = binSums[i] / binCounts[i];
            else if(i > 0)
                erf[i] = 0;
        }
    }
    
    // 插值处理空缺数据
    void interpolateERF() {
        for (int i = 0; i < numBins; i++) {
            if (binCounts[i] == 0) {
             
                int prev = i - 1;
                while (prev >= 0 && binCounts[prev] == 0) prev--;
                
              
                int next = i + 1;
                while (next < numBins && binCounts[next] == 0) next++;
        
          
                if (prev >= 0 && next < numBins) {
                 
                    double alpha = static_cast<double>(i - prev) / (next - prev);
                    erf[i] = erf[prev] * (1 - alpha) + erf[next] * alpha;
                } else if (prev >= 0) {
             
                    erf[i] = erf[prev];
                } else if (next < numBins) {
                
                    erf[i] = erf[next];
                } else {
                  
                    erf[i] = 0.0;
                }
            }
        }
    }
    
    // 三次多项式平滑处理
    void smoothERF(int windowSize = 21) {
        int halfWin = windowSize / 2;
        erfSmoothed.clear();
        
        // 对于无法构成完整窗口的首尾数据，将舍去
        for (int i = 0; i <= numBins - windowSize; i++) {
            double fittedVal = cubicFitAtCenter(erf, i, windowSize);
            erfSmoothed.push_back(fittedVal);
        }
    }
    
    // 完整的ERF计算流程
    void calculateERF(const Mat& image, int windowSize = 21) {
        calculateInitialERF(image);
        interpolateERF();
        smoothERF(windowSize);
    }
    
    // 获取原始ERF数据
    const std::vector<double>& getERF() const {
        return erf;
    }
    
   
    // 获取平滑后的ERF数据
    const std::vector<double>& getSmoothedERF() const {
        return erfSmoothed;
    }
    // 获取ERF参数信息
    float getBinSize() const { return binSize; }
    float getInnerRadius() const { return innerRadius; }
    float getOuterRadius() const { return outerRadius; }
    int getNumBins() const { return numBins; }
    std::vector<int> getBinCounts() const { return binCounts; }
    // 输出ERF数据
    void printSmoothedERF() const {
        cout << "Radius\tSmoothed ERF" << endl;
        int halfWin = 21 / 2; // 默认窗口大小的一半
        for (size_t i = 0; i < erfSmoothed.size(); i++) {
            float radius = innerRadius + (i + halfWin) * binSize;
            cout << radius << "\t" << erfSmoothed[i] << endl;
        }
    }
    std::vector<double> getErfSmoothed() const {
        return erfSmoothed;
    }
};

//plot函数
template <typename T>
cv::Mat plotVector(const std::vector<T>& data, int width = 1000, int height = 600, float binSize = 0.05) {
  
    if (data.size() <= 1) {
        cv::Mat errorImage = cv::Mat::zeros(height, width, CV_8UC3);
        cv::putText(errorImage, "Data vector has <= 1 element.", cv::Point(20, height / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        return errorImage;
    }

   
    cv::Mat plotImage = cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    // 定义图表区域的边距和尺寸
    int margin = 50;
    int graphWidth = width - 4 * margin;
    int graphHeight = height - 2 * margin;
    // 图表区域的左上角坐标
    int graphX = 2 * margin;
    int graphY = margin;

    // 寻找数据的最大值和最小值，用于Y轴的缩放
    auto minmax_it = std::minmax_element(data.begin(), data.end());
    T minVal = *minmax_it.first;
    T maxVal = *minmax_it.second;

    if (maxVal == minVal) {
        maxVal += static_cast<T>(1.0);
    }

    // 绘制背景网格和坐标轴
    std::string title = "Custom Plot";
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(title, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
    int textX = (width - textSize.width) / 2;
    int textY = margin / 2 + textSize.height / 2;
    cv::putText(plotImage, title, cv::Point(textX, textY), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);

 
    int numGridLines = 5;

    for (int i = 0; i <= numGridLines; ++i) {
        int y = graphY + (i * graphHeight / numGridLines);
        cv::line(plotImage, cv::Point(graphX, y), cv::Point(graphX + graphWidth, y), cv::Scalar(220, 220, 220));
    }

    for (int i = 0; i <= numGridLines; ++i) {
        int x = graphX + (i * graphWidth / numGridLines);
        cv::line(plotImage, cv::Point(x, graphY), cv::Point(x, graphY + graphHeight), cv::Scalar(220, 220, 220));
    }
 
    cv::rectangle(plotImage, cv::Point(graphX, graphY), cv::Point(graphX + graphWidth, graphY + graphHeight), cv::Scalar(0, 0, 0), 1);

    // 绘制数据点之间的连线
    std::vector<cv::Point> plotPoints;
    for (size_t i = 0; i < data.size(); ++i) {
     
        int px = graphX + static_cast<int>((i / (double)(data.size() - 1)) * graphWidth);

        int py = graphY + graphHeight - static_cast<int>(((data[i] - minVal) / (maxVal - minVal)) * graphHeight);

        plotPoints.push_back(cv::Point(px, py));
    }
   
    cv::polylines(plotImage, plotPoints, false, cv::Scalar(255, 0, 0), 2); // 蓝色线条

    //  绘制坐标轴上的标签，

    cv::putText(plotImage, cv::format("%.1f", static_cast<double>(maxVal)), cv::Point(margin - 45, margin + 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
    cv::putText(plotImage, cv::format("%.1f", static_cast<double>(minVal)), cv::Point(margin - 45, margin + graphHeight), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));


    cv::putText(plotImage, "0", cv::Point(margin, margin + graphHeight + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
    cv::putText(plotImage, std::to_string((data.size() - 1) * binSize), cv::Point(width - margin - 20, margin + graphHeight + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));

    return plotImage;
}

// Base64 编码函数
std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];
    static const std::string base64_chars =
                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                 "abcdefghijklmnopqrstuvwxyz"
                 "0123456789+/";
    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            for(i = 0; (i <4) ; i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }
    if (i)
    {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0';
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;
        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];
        while((i++ < 3))
            ret += '=';
    }
    return ret;
}

// 将 cv::Mat 转换为 Base64 字符串
std::string matToBase64(const cv::Mat& mat) {
    std::vector<uchar> buf;
    cv::imencode(".png", mat, buf);
    auto* enc_msg = reinterpret_cast<unsigned char*>(buf.data());
    std::string base64_png = base64_encode(enc_msg, buf.size());
    return base64_png;
}



// ----------------------------------------------------------------------------------------
//拟合函数模型
double model(
    const input_vector& input,
    const parameter_vector& params
)
{
    const double u = params(0);
    const double A = params(1);
    const double B = params(2);
    const double sigma = params(3);

    const double x = input(0);

    const double temp = A * exp(-pow((x - u) / sigma, 2) / 2) / (sqrt(2 * pi) * sigma) + B;

    return temp;
}

// ----------------------------------------------------------------------------------------

//残差计算函数
double residual(
    const std::pair<input_vector, double>& data,
    const parameter_vector& params
)
{
    return model(data.first, params) - data.second;
}

// ----------------------------------------------------------------------------------------

// 雅可比矩阵计算函数
parameter_vector residual_derivative(
    const std::pair<input_vector, double>& data,
    const parameter_vector& params
)
{
    parameter_vector der;

    const double u = params(0);
    const double A = params(1);
    const double B = params(2);
    const double sigma = params(3);

    const double x = data.first(0);


    const double temp = A * exp(-pow((x - u) / sigma, 2) / 2) / (sqrt(2 * pi) * sigma);

    der(0) = ((x - u) / pow(sigma, 2)) * temp;
    der(1) = exp(-pow((x - u) / sigma, 2) / 2) / (sqrt(2 * pi) * sigma);
    der(2) = 1;
    der(3) = -temp / sigma + (pow((x - u), 2) / pow(sigma, 3)) * temp;


    return der;
}

//关闭 cout 输出
void disableCout() {
    std::cout.setstate(std::ios_base::failbit);
}

// 恢复 cout 输出
void enableCout() {
    std::cout.clear();
}

int main(int argc, char* argv[]) {
    std::string csr_filename;
    disableCout(); // 关闭 cout 输出

    if (argc < 2) {
        
        csr_filename = "E:\\programming\\projects\\SRT\\Project1\\CTTestImage1.csr";
        std::cout << "No input file provided. Using default: " << csr_filename << std::endl;
    } else {
        
        csr_filename = argv[1];
    }

    try {
        // 执行核心的计算逻辑 
        double binSize = 0.05; // 每个bin的宽度，单位像素
        CircleDetector detector(csr_filename);
        detector.detectCircles();
        int circle_windows = 40;
        ERFCalculator erfCalc(detector, binSize, circle_windows);
        erfCalc.calculateERF(detector.original_data, 21);
    

        // 获取平滑后的ERF数据
        std::vector<double> erf_smoothed = erfCalc.getSmoothedERF();
        std::vector<double> erf_initial = erfCalc.getERF();
        // 计算PSF
        std::vector<double> psf = calculatePSF(erf_smoothed, 21);
        std::vector<double> psf_Gaussian;
        std::vector<double> mtf;
        // dlib库进行高斯函数拟合
        try
        {

            std::vector<std::pair<input_vector, double> > data_samples;
            input_vector input;
            parameter_vector params;
          
            params(0) = std::distance(psf.begin(), std::min_element(psf.begin(), psf.end())); 
            params(1) = *std::min_element(psf.begin(), psf.end()); 
            params(2) = *std::max_element(psf.begin(), psf.end()); 
            params(3) = 100.0; 
            for (int i = 0; i < psf.size(); ++i)
            {
                input = i;
                data_samples.push_back(make_pair(input, psf[i]));
            }
            solve_least_squares_lm(objective_delta_stop_strategy(1e-7).be_verbose(),
                residual,
                residual_derivative,
                data_samples,
                params);
            enableCout(); // 恢复 cout 输出
            for (int i = 0; i < psf.size(); i++) {
                input = i;
                psf_Gaussian.push_back(model(input, params));
            }
			//计算MTF
            mtf = calculateMTF(params, binSize);
        }
        catch (std::exception& e)
        {
            std::cerr << e.what() << endl;
        }
        
        
        // 将计算结果打包成JSON对象
        json result_json;
        result_json["circle"] = detector.circles[0][0];
        result_json["Image"] = matToBase64(detector.Image);
        result_json["binSize"] = binSize;
		result_json["erf"] = erf_smoothed;
        result_json["psf"] = psf_Gaussian;
        result_json["mtf"] = mtf;
        std::cout << result_json.dump() << std::endl;
        disableCout(); // 关闭 cout 输出
		

		
        
        

    } catch (const std::exception& e) {
        // 捕获处理过程中可能发生的任何异常 (如文件打不开、未检测到圆等)
        // 将错误信息输出到标准错误(stderr)
        std::cerr << "An error occurred during processing: " << e.what() << std::endl;
        return 1; 
    }
     
    // 程序成功运行，返回0
    return 0;
}








