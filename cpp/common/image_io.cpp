// cpp/common/image_io.cpp

#include "image_io.hpp"

#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <cstring>

namespace gcv
{

    Image make_image(int width, int height, int channels)
    {
        Image img;
        img.width = width;
        img.height = height;
        img.channels = channels;
        img.data = std::make_unique<uint8_t[]>(img.size_bytes());
        return img;
    }

    Image load_image_rgb(const std::string &path)
    {
        // OpenCV loads as BGR by default; convert to RGB so our math (and the
        // Gray = 0.299R + 0.587G + 0.114B formula from the SRS) matches the
        // channel order we document everywhere else.
        cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
        if (bgr.empty())
        {
            throw std::runtime_error("Failed to read image: " + path);
        }

        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

        Image img = make_image(rgb.cols, rgb.rows, 3);
        std::memcpy(img.data.get(), rgb.data, img.size_bytes());
        return img;
    }

    void save_image(const std::string &path, const Image &img)
    {
        if (img.channels == 3)
        {
            cv::Mat rgb(img.height, img.width, CV_8UC3, img.data.get());
            cv::Mat bgr;
            cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
            if (!cv::imwrite(path, bgr))
            {
                throw std::runtime_error("Failed to write image: " + path);
            }
        }
        else if (img.channels == 1)
        {
            cv::Mat gray(img.height, img.width, CV_8UC1, img.data.get());
            if (!cv::imwrite(path, gray))
            {
                throw std::runtime_error("Failed to write image: " + path);
            }
        }
        else
        {
            throw std::runtime_error("save_image: unsupported channel count " +
                                     std::to_string(img.channels));
        }
    }

    Image resize_image(const Image &img, int new_width, int new_height)
    {
        if (img.channels != 1 && img.channels != 3)
        {
            throw std::runtime_error("resize_image: unsupported channel count " +
                                     std::to_string(img.channels));
        }

        const int cv_type = (img.channels == 3) ? CV_8UC3 : CV_8UC1;
        cv::Mat src(img.height, img.width, cv_type, const_cast<uint8_t *>(img.data.get()));

        cv::Mat resized;
        cv::resize(src, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

        Image out = make_image(new_width, new_height, img.channels);
        std::memcpy(out.data.get(), resized.data, out.size_bytes());
        return out;
    }

} // namespace gcv