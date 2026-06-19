#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// =========================================================
// CONFIG
// =========================================================

// Detection model path. This ONNX model finds text boxes from the full image.
static const std::string DET_MODEL = "models_onnx/PP-OCRv5_server_det_infer.onnx";

// Recognition model path. This ONNX model reads text from each cropped text box.
static const std::string REC_MODEL = "models_onnx/korean_PP-OCRv5_mobile_rec.onnx";

// PaddleOCR dictionary path. Index 0 is CTC blank; real characters start from index 1.
static const std::string REC_DICT = "korean_dict.txt";

// Test image directory. Every supported image in this directory will be processed.
static const std::string TEST_IMAGE_DIR = "./data";

// Output directory for JSON, visualization images, debug bitmap, and crops.
static const std::string OUTPUT_DIR = "./recognition_results";

// Set true to save JSON/images/crops. Set false to print JSON only in terminal.
static const bool SAVE_OUTPUTS = true;

// Save cropped text line images for debugging detector and recognizer alignment.
static const bool SAVE_CROPS = true;

// Print raw recognizer output for every detected crop while SAVE_OUTPUTS is true.
static const bool DEBUG_RAW_REC = true;

// PaddleOCR DB detector postprocess values.
static const float DET_THRESH = 0.30f;
static const float BOX_THRESH = 0.30f;
static const float UNCLIP_RATIO = 1.7f;
static const int DET_LIMIT_SIDE_LEN = 1536;

// PaddleOCR recognition resize values.
static const int DEFAULT_REC_HEIGHT = 48;
static const int MAX_DYNAMIC_REC_WIDTH = 2048;

// =========================================================
// STRUCTS
// =========================================================

struct DetItem {
    std::vector<cv::Point2f> box;
    float det_score = 0.0f;
};

struct OCRResult {
    std::string text;
    float rec_score = 0.0f;
    float det_score = 0.0f;
    std::vector<cv::Point2f> box;
};

struct TensorData {
    std::vector<float> values;
    std::vector<int64_t> shape;
};

struct OnnxModel {
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::string input_name;
    std::string output_name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;

    OnnxModel(Ort::Env& env, const std::string& model_path)
        : session(nullptr) {
        // ONNX Runtime CPU provider keeps deployment light and avoids Paddle dependencies.
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session(env, model_path.c_str(), options);

        input_name = session.GetInputNameAllocated(0, allocator).get();
        output_name = session.GetOutputNameAllocated(0, allocator).get();

        Ort::TypeInfo input_info = session.GetInputTypeInfo(0);
        Ort::TypeInfo output_info = session.GetOutputTypeInfo(0);

        input_shape = input_info.GetTensorTypeAndShapeInfo().GetShape();
        output_shape = output_info.GetTensorTypeAndShapeInfo().GetShape();
    }

    Ort::Value run(const TensorData& input) {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input.values.data()),
            input.values.size(),
            input.shape.data(),
            input.shape.size()
        );

        const char* input_names[] = {input_name.c_str()};
        const char* output_names[] = {output_name.c_str()};

        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        return std::move(outputs.front());
    }
};

// =========================================================
// UTILS
// =========================================================

std::string getStem(const std::string& path) {
    return fs::path(path).stem().string();
}

std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);

    for (unsigned char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                out += static_cast<char>(c);
                break;
        }
    }

    return out;
}

bool containsKorean(const std::string& utf8) {
    // Simple UTF-8 range check for Hangul syllables.
    // Hangul syllables: U+AC00 - U+D7A3.
    const unsigned char* s = reinterpret_cast<const unsigned char*>(utf8.c_str());

    while (*s) {
        uint32_t cp = 0;

        if ((*s & 0x80) == 0) {
            cp = *s;
            s += 1;
        } else if ((*s & 0xE0) == 0xC0) {
            cp = ((*s & 0x1F) << 6) | (*(s + 1) & 0x3F);
            s += 2;
        } else if ((*s & 0xF0) == 0xE0) {
            cp = ((*s & 0x0F) << 12) | ((*(s + 1) & 0x3F) << 6) | (*(s + 2) & 0x3F);
            s += 3;
        } else if ((*s & 0xF8) == 0xF0) {
            cp = ((*s & 0x07) << 18) | ((*(s + 1) & 0x3F) << 12) |
                 ((*(s + 2) & 0x3F) << 6) | (*(s + 3) & 0x3F);
            s += 4;
        } else {
            s += 1;
            continue;
        }

        if (cp >= 0xAC00 && cp <= 0xD7A3) {
            return true;
        }
    }

    return false;
}

cv::Rect safeRect(int x1, int y1, int x2, int y2, int w, int h) {
    x1 = std::max(0, std::min(x1, w - 1));
    y1 = std::max(0, std::min(y1, h - 1));
    x2 = std::max(0, std::min(x2, w - 1));
    y2 = std::max(0, std::min(y2, h - 1));

    if (x2 < x1) std::swap(x1, x2);
    if (y2 < y1) std::swap(y1, y2);

    return cv::Rect(x1, y1, std::max(1, x2 - x1 + 1), std::max(1, y2 - y1 + 1));
}

bool isImageFile(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
           ext == ".bmp" || ext == ".tif" || ext == ".tiff" ||
           ext == ".webp";
}

std::vector<std::string> getImagePaths(const std::string& image_dir) {
    fs::path input(image_dir);
    std::vector<std::string> paths;

    if (fs::is_regular_file(input) && isImageFile(input)) {
        paths.push_back(input.string());
        return paths;
    }

    if (!fs::exists(input)) {
        throw std::runtime_error("Image path not found: " + image_dir);
    }

    for (const auto& item : fs::directory_iterator(input)) {
        if (fs::is_regular_file(item.path()) && isImageFile(item.path())) {
            paths.push_back(item.path().string());
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

// =========================================================
// DICTIONARY
// =========================================================

std::vector<std::string> loadDict(const std::string& path) {
    if (!fs::exists(path)) {
        throw std::runtime_error("Dictionary not found: " + path);
    }

    std::vector<std::string> chars;
    chars.push_back("blank");

    std::ifstream f(path);
    std::string line;

    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (!line.empty()) {
            chars.push_back(line);
        }
    }

    if (chars.empty() || chars.back() != " ") {
        chars.push_back(" ");
    }

    if (SAVE_OUTPUTS) {
        std::cout << "Dictionary classes: " << chars.size() << std::endl;
    }
    return chars;
}

// =========================================================
// PREPROCESSING
// =========================================================

cv::Mat resizeDet(const cv::Mat& rgb, int limit_side_len = DET_LIMIT_SIDE_LEN) {
    int h = rgb.rows;
    int w = rgb.cols;

    float ratio = std::min(limit_side_len / static_cast<float>(std::max(h, w)), 1.0f);

    int resize_h = static_cast<int>(std::round(h * ratio / 32.0f) * 32);
    int resize_w = static_cast<int>(std::round(w * ratio / 32.0f) * 32);

    resize_h = std::max(32, resize_h);
    resize_w = std::max(32, resize_w);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(resize_w, resize_h));

    return resized;
}

TensorData matToDetTensor(const cv::Mat& rgb) {
    int h = rgb.rows;
    int w = rgb.cols;

    TensorData tensor;
    tensor.shape = {1, 3, h, w};
    tensor.values.resize(static_cast<size_t>(1) * 3 * h * w);

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};

    for (int y = 0; y < h; ++y) {
        const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);

        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < 3; ++c) {
                float v = row[x][c] / 255.0f;
                v = (v - mean[c]) / stdv[c];

                size_t idx = static_cast<size_t>(c) * h * w + static_cast<size_t>(y) * w + x;
                tensor.values[idx] = v;
            }
        }
    }

    return tensor;
}

TensorData matToRecTensor(const cv::Mat& rgb) {
    int h = rgb.rows;
    int w = rgb.cols;

    TensorData tensor;
    tensor.shape = {1, 3, h, w};
    tensor.values.resize(static_cast<size_t>(1) * 3 * h * w);

    for (int y = 0; y < h; ++y) {
        const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);

        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < 3; ++c) {
                float v = row[x][c] / 255.0f;
                v = (v - 0.5f) / 0.5f;

                size_t idx = static_cast<size_t>(c) * h * w + static_cast<size_t>(y) * w + x;
                tensor.values[idx] = v;
            }
        }
    }

    return tensor;
}

// =========================================================
// DETECTION POSTPROCESSING
// =========================================================

std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point2f>& pts) {
    std::vector<cv::Point2f> rect(4);

    std::vector<float> sums(4);
    std::vector<float> diffs(4);

    for (int i = 0; i < 4; ++i) {
        sums[i] = pts[i].x + pts[i].y;
        diffs[i] = pts[i].y - pts[i].x;
    }

    int tl = std::min_element(sums.begin(), sums.end()) - sums.begin();
    int br = std::max_element(sums.begin(), sums.end()) - sums.begin();
    int tr = std::min_element(diffs.begin(), diffs.end()) - diffs.begin();
    int bl = std::max_element(diffs.begin(), diffs.end()) - diffs.begin();

    rect[0] = pts[tl];
    rect[1] = pts[tr];
    rect[2] = pts[br];
    rect[3] = pts[bl];

    return rect;
}

std::pair<std::vector<cv::Point2f>, float> getMiniBox(const std::vector<cv::Point>& contour) {
    cv::RotatedRect rr = cv::minAreaRect(contour);

    cv::Point2f pts_arr[4];
    rr.points(pts_arr);

    std::vector<cv::Point2f> pts(pts_arr, pts_arr + 4);
    pts = orderPoints(pts);

    float side1 = cv::norm(pts[0] - pts[1]);
    float side2 = cv::norm(pts[1] - pts[2]);

    return {pts, std::min(side1, side2)};
}

std::pair<std::vector<cv::Point2f>, float> getMiniBoxF(const std::vector<cv::Point2f>& contourf) {
    std::vector<cv::Point> contour;
    contour.reserve(contourf.size());

    for (const auto& p : contourf) {
        contour.emplace_back(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y)));
    }

    return getMiniBox(contour);
}

float boxScoreFast(const cv::Mat& prob, const std::vector<cv::Point2f>& box) {
    int h = prob.rows;
    int w = prob.cols;

    float min_x = box[0].x, max_x = box[0].x;
    float min_y = box[0].y, max_y = box[0].y;

    for (const auto& p : box) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }

    int xmin = std::max(0, static_cast<int>(std::floor(min_x)));
    int xmax = std::min(w - 1, static_cast<int>(std::ceil(max_x)));
    int ymin = std::max(0, static_cast<int>(std::floor(min_y)));
    int ymax = std::min(h - 1, static_cast<int>(std::ceil(max_y)));

    if (xmax <= xmin || ymax <= ymin) {
        return 0.0f;
    }

    cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

    std::vector<cv::Point> local_box;
    for (const auto& p : box) {
        local_box.emplace_back(
            static_cast<int>(std::round(p.x - xmin)),
            static_cast<int>(std::round(p.y - ymin))
        );
    }

    std::vector<std::vector<cv::Point>> polys = {local_box};
    cv::fillPoly(mask, polys, cv::Scalar(1));

    cv::Mat roi = prob(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
    return static_cast<float>(cv::mean(roi, mask)[0]);
}

// Approximation of pyclipper unclip.
// For exact PaddleOCR-style unclip, use Clipper2 C++.
std::vector<cv::Point2f> unclipApprox(const std::vector<cv::Point2f>& box, float ratio = UNCLIP_RATIO) {
    cv::Point2f center(0, 0);

    for (const auto& p : box) {
        center += p;
    }

    center.x /= static_cast<float>(box.size());
    center.y /= static_cast<float>(box.size());

    std::vector<cv::Point2f> expanded;
    expanded.reserve(box.size());

    for (const auto& p : box) {
        cv::Point2f v = p - center;
        expanded.push_back(center + v * ratio);
    }

    return expanded;
}

std::vector<DetItem> detect(OnnxModel& det_model,
                            const cv::Mat& rgb,
                            const std::string& debug_bitmap_path) {
    cv::Mat resized = resizeDet(rgb);

    TensorData input_tensor = matToDetTensor(resized);
    Ort::Value output = det_model.run(input_tensor);

    auto info = output.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();

    if (shape.size() != 4) {
        throw std::runtime_error("Detection output shape is not 4D.");
    }

    int prob_h = static_cast<int>(shape[2]);
    int prob_w = static_cast<int>(shape[3]);

    const float* out_data = output.GetTensorData<float>();

    cv::Mat prob(prob_h, prob_w, CV_32FC1);
    std::memcpy(prob.data, out_data, sizeof(float) * prob_h * prob_w);

    cv::Mat bitmap;
    cv::threshold(prob, bitmap, DET_THRESH, 255, cv::THRESH_BINARY);
    bitmap.convertTo(bitmap, CV_8UC1);

    if (!debug_bitmap_path.empty()) {
        cv::imwrite(debug_bitmap_path, bitmap);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int src_h = rgb.rows;
    int src_w = rgb.cols;

    std::vector<DetItem> boxes;

    for (const auto& contour : contours) {
        if (contour.size() < 3) {
            continue;
        }

        auto [box, short_side] = getMiniBox(contour);

        if (short_side < 3.0f) {
            continue;
        }

        float score = boxScoreFast(prob, box);

        if (score < BOX_THRESH) {
            continue;
        }

        std::vector<cv::Point2f> expanded = unclipApprox(box);

        auto [box2, short_side2] = getMiniBoxF(expanded);

        if (short_side2 < 3.0f) {
            continue;
        }

        for (auto& p : box2) {
            p.x = std::round(p.x / prob_w * src_w);
            p.y = std::round(p.y / prob_h * src_h);

            p.x = std::max(0.0f, std::min(p.x, static_cast<float>(src_w - 1)));
            p.y = std::max(0.0f, std::min(p.y, static_cast<float>(src_h - 1)));
        }

        box2 = orderPoints(box2);

        boxes.push_back({box2, score});
    }

    std::sort(boxes.begin(), boxes.end(), [](const DetItem& a, const DetItem& b) {
        float ay = std::min({a.box[0].y, a.box[1].y, a.box[2].y, a.box[3].y});
        float by = std::min({b.box[0].y, b.box[1].y, b.box[2].y, b.box[3].y});

        if (std::abs(ay - by) > 5.0f) {
            return ay < by;
        }

        float ax = std::min({a.box[0].x, a.box[1].x, a.box[2].x, a.box[3].x});
        float bx = std::min({b.box[0].x, b.box[1].x, b.box[2].x, b.box[3].x});

        return ax < bx;
    });

    return boxes;
}

// =========================================================
// CROP
// =========================================================

cv::Mat cropText(const cv::Mat& rgb, const std::vector<cv::Point2f>& input_box) {
    std::vector<cv::Point2f> box = orderPoints(input_box);

    int crop_w = static_cast<int>(std::max(
        cv::norm(box[0] - box[1]),
        cv::norm(box[2] - box[3])
    ));

    int crop_h = static_cast<int>(std::max(
        cv::norm(box[0] - box[3]),
        cv::norm(box[1] - box[2])
    ));

    crop_w = std::max(1, crop_w);
    crop_h = std::max(1, crop_h);

    std::vector<cv::Point2f> dst = {
        {0.0f, 0.0f},
        {static_cast<float>(crop_w), 0.0f},
        {static_cast<float>(crop_w), static_cast<float>(crop_h)},
        {0.0f, static_cast<float>(crop_h)}
    };

    cv::Mat m = cv::getPerspectiveTransform(box, dst);

    cv::Mat crop;
    cv::warpPerspective(
        rgb,
        crop,
        m,
        cv::Size(crop_w, crop_h),
        cv::INTER_CUBIC,
        cv::BORDER_REPLICATE
    );

    if (crop.rows / static_cast<float>(std::max(1, crop.cols)) >= 1.5f) {
        cv::rotate(crop, crop, cv::ROTATE_90_COUNTERCLOCKWISE);
    }

    return crop;
}

// =========================================================
// RECOGNITION
// =========================================================

int getRecHeight(const OnnxModel& rec_model) {
    if (rec_model.input_shape.size() >= 4 && rec_model.input_shape[2] > 0) {
        return static_cast<int>(rec_model.input_shape[2]);
    }

    return DEFAULT_REC_HEIGHT;
}

cv::Mat resizeRec(const cv::Mat& crop_rgb,
                  int rec_h = DEFAULT_REC_HEIGHT,
                  int max_dynamic_w = MAX_DYNAMIC_REC_WIDTH) {
    int h = crop_rgb.rows;
    int w = crop_rgb.cols;

    if (h <= 0 || w <= 0) {
        return cv::Mat();
    }

    float ratio = w / static_cast<float>(h);
    int new_w = static_cast<int>(std::ceil(rec_h * ratio));

    int final_w = std::min(max_dynamic_w, std::max(32, new_w));

    new_w = std::min(new_w, final_w);
    new_w = std::max(16, new_w);

    cv::Mat resized;
    cv::resize(crop_rgb, resized, cv::Size(new_w, rec_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat padded(rec_h, final_w, CV_8UC3, cv::Scalar(255, 255, 255));
    resized.copyTo(padded(cv::Rect(0, 0, new_w, rec_h)));

    return padded;
}

std::pair<std::string, float> decodeCTC(const Ort::Value& output,
                                        const std::vector<std::string>& chars) {
    auto info = output.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();
    const float* data = output.GetTensorData<float>();

    size_t T = 0;
    size_t C = 0;
    size_t offset = 0;
    bool transposed = false;

    if (shape.size() == 3) {
        // Usually [1, T, C], but some exports can return [1, C, T].
        if (shape[1] == static_cast<int64_t>(chars.size())) {
            C = shape[1];
            T = shape[2];
            transposed = true;
        } else {
            T = shape[1];
            C = shape[2];
        }
    } else if (shape.size() == 2) {
        if (shape[0] == static_cast<int64_t>(chars.size())) {
            C = shape[0];
            T = shape[1];
            transposed = true;
        } else {
            T = shape[0];
            C = shape[1];
        }
    } else {
        throw std::runtime_error("Recognition output shape must be 2D or 3D.");
    }

    if (SAVE_OUTPUTS && C != chars.size()) {
        std::cout << "WARNING: dict classes=" << chars.size()
                  << ", model classes=" << C << std::endl;
    }

    std::string text;
    std::vector<float> scores;

    int last_idx = -1;

    for (size_t t = 0; t < T; ++t) {
        int best_idx = 0;
        float best_val = transposed ? data[offset + t] : data[offset + t * C];

        for (size_t c = 1; c < C; ++c) {
            float value = transposed ? data[offset + c * T + t] : data[offset + t * C + c];

            if (value > best_val) {
                best_val = value;
                best_idx = static_cast<int>(c);
            }
        }

        if (best_idx != 0 && best_idx != last_idx && best_idx < static_cast<int>(chars.size())) {
            text += chars[best_idx];
            scores.push_back(best_val);
        }

        last_idx = best_idx;
    }

    float score = 0.0f;

    if (!scores.empty()) {
        score = std::accumulate(scores.begin(), scores.end(), 0.0f) / scores.size();
    }

    return {text, score};
}

std::pair<std::string, float> recognize(OnnxModel& rec_model,
                                        const cv::Mat& crop_rgb,
                                        const std::vector<std::string>& chars) {
    int rec_h = getRecHeight(rec_model);

    cv::Mat rec_img = resizeRec(crop_rgb, rec_h, MAX_DYNAMIC_REC_WIDTH);

    if (rec_img.empty()) {
        return {"", 0.0f};
    }

    TensorData input_tensor = matToRecTensor(rec_img);
    Ort::Value output = rec_model.run(input_tensor);

    return decodeCTC(output, chars);
}

// =========================================================
// JSON SAVE
// =========================================================

std::string buildImageJson(const std::string& image_path,
                           const std::string& det_model,
                           const std::string& rec_model,
                           const std::vector<OCRResult>& results,
                           int indent = 0) {
    std::ostringstream f;
    std::string pad(indent, ' ');
    std::string pad2(indent + 2, ' ');
    std::string pad4(indent + 4, ' ');
    std::string pad6(indent + 6, ' ');

    f << pad << "{\n";
    f << pad2 << "\"image\": \"" << jsonEscape(image_path) << "\",\n";
    f << pad2 << "\"det_model\": \"" << jsonEscape(det_model) << "\",\n";
    f << pad2 << "\"rec_model\": \"" << jsonEscape(rec_model) << "\",\n";
    f << pad2 << "\"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];

        f << pad4 << "{\n";
        f << pad6 << "\"text\": \"" << jsonEscape(r.text) << "\",\n";
        f << pad6 << "\"rec_score\": " << r.rec_score << ",\n";
        f << pad6 << "\"det_score\": " << r.det_score << ",\n";
        f << pad6 << "\"box\": [";

        for (size_t j = 0; j < r.box.size(); ++j) {
            f << "[" << r.box[j].x << ", " << r.box[j].y << "]";
            if (j + 1 < r.box.size()) f << ", ";
        }

        f << "]\n";
        f << pad4 << "}";

        if (i + 1 < results.size()) {
            f << ",";
        }

        f << "\n";
    }

    f << pad2 << "]\n";
    f << pad << "}";

    return f.str();
}

void saveJson(const std::string& json_path,
              const std::string& image_path,
              const std::string& det_model,
              const std::string& rec_model,
              const std::vector<OCRResult>& results) {
    std::ofstream f(json_path);
    f << buildImageJson(image_path, det_model, rec_model, results) << "\n";
}

// =========================================================
// VISUALIZATION
// =========================================================

std::vector<cv::Scalar> paletteBGR() {
    return {
        cv::Scalar(121, 236, 202),  // light green
        cv::Scalar(196, 230, 125),  // mint
        cv::Scalar(226, 161, 185),  // purple
        cv::Scalar(255, 203, 134),  // blue
        cv::Scalar(102, 180, 246),  // orange
        cv::Scalar(190, 245, 152),  // green
        cv::Scalar(230, 225, 210),  // grey-blue
        cv::Scalar(120, 225, 255)   // yellow
    };
}

cv::Rect polygonBBox(const std::vector<cv::Point2f>& box, int img_w, int img_h) {
    float min_x = box[0].x, max_x = box[0].x;
    float min_y = box[0].y, max_y = box[0].y;

    for (const auto& p : box) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }

    return safeRect(
        static_cast<int>(std::floor(min_x)),
        static_cast<int>(std::floor(min_y)),
        static_cast<int>(std::ceil(max_x)),
        static_cast<int>(std::ceil(max_y)),
        img_w,
        img_h
    );
}

void blendRect(cv::Mat& img, const cv::Rect& rect, const cv::Scalar& color, double alpha) {
    cv::Rect r = rect & cv::Rect(0, 0, img.cols, img.rows);

    if (r.empty()) {
        return;
    }

    cv::Mat roi = img(r);
    cv::Mat color_mat(roi.size(), roi.type(), color);

    cv::addWeighted(color_mat, alpha, roi, 1.0 - alpha, 0.0, roi);
    cv::rectangle(img, r, color, 1);
}

void drawSideBySide(const std::string& image_path,
                    const std::vector<OCRResult>& results,
                    const std::string& save_path) {
    cv::Mat left_bgr = cv::imread(image_path);

    if (left_bgr.empty()) {
        throw std::runtime_error("Image not found: " + image_path);
    }

    int img_w = left_bgr.cols;
    int img_h = left_bgr.rows;
    int gap = 30;
    int right_w = img_w;

    cv::Mat canvas(img_h, img_w + gap + right_w, CV_8UC3, cv::Scalar(255, 255, 255));

    left_bgr.copyTo(canvas(cv::Rect(0, 0, img_w, img_h)));

    int rx = img_w + gap;

    cv::line(
        canvas,
        cv::Point(img_w + gap / 2, 0),
        cv::Point(img_w + gap / 2, img_h),
        cv::Scalar(220, 220, 220),
        1
    );

    auto colors = paletteBGR();

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];

        cv::Scalar color = colors[i % colors.size()];

        cv::Rect bbox = polygonBBox(r.box, img_w, img_h);

        cv::Rect left_rect = safeRect(
            bbox.x - 2,
            bbox.y - 2,
            bbox.x + bbox.width + 2,
            bbox.y + bbox.height + 2,
            img_w,
            img_h
        );

        // LEFT: color overlay on original text region
        cv::Rect left_canvas_rect = left_rect;
        blendRect(canvas, left_canvas_rect, color, 0.42);

        int right_x1 = rx + bbox.x;
        int right_y1 = std::max(0, bbox.y - 2);

        int right_box_w = std::max(1, bbox.width);
        int right_box_h = std::max(1, bbox.height);

        bool use_crop_for_text = containsKorean(r.text);

        cv::Rect right_rect;

        if (use_crop_for_text) {
            right_rect = safeRect(
                right_x1,
                right_y1,
                right_x1 + right_box_w + 8,
                right_y1 + right_box_h + 5,
                canvas.cols,
                canvas.rows
            );
        } else {
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(
                r.text,
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                1,
                &baseline
            );

            right_rect = safeRect(
                right_x1,
                right_y1,
                right_x1 + text_size.width + 8,
                right_y1 + std::max(right_box_h, text_size.height + 6),
                canvas.cols,
                canvas.rows
            );
        }

        blendRect(canvas, right_rect, color, 0.28);

        if (use_crop_for_text) {
            cv::Rect crop_rect = bbox & cv::Rect(0, 0, img_w, img_h);

            if (!crop_rect.empty()) {
                cv::Mat crop = left_bgr(crop_rect).clone();

                int target_w = std::max(1, right_rect.width - 6);
                int target_h = std::max(1, right_rect.height - 4);

                cv::resize(crop, crop, cv::Size(target_w, target_h), 0, 0, cv::INTER_LANCZOS4);

                cv::Rect paste_rect(
                    right_rect.x + 3,
                    right_rect.y + 2,
                    std::min(target_w, canvas.cols - right_rect.x - 3),
                    std::min(target_h, canvas.rows - right_rect.y - 2)
                );

                if (!paste_rect.empty()) {
                    crop(cv::Rect(0, 0, paste_rect.width, paste_rect.height)).copyTo(canvas(paste_rect));
                }
            }
        } else {
            cv::putText(
                canvas,
                r.text,
                cv::Point(right_rect.x + 4, right_rect.y + right_rect.height - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(0, 0, 0),
                1,
                cv::LINE_AA
            );
        }
    }

    cv::imwrite(save_path, canvas);
}

// =========================================================
// IMAGE PIPELINE
// =========================================================

std::vector<OCRResult> processImage(const std::string& image_path,
                                    OnnxModel& det_model,
                                    OnnxModel& rec_model,
                                    const std::vector<std::string>& chars) {
    std::string image_stem = getStem(image_path);
    std::string crop_dir = OUTPUT_DIR + "/" + image_stem + "_onnx_debug_crops";
    std::string debug_bitmap_path = SAVE_OUTPUTS ? OUTPUT_DIR + "/" + image_stem + "_onnx_debug_bitmap.png" : "";

    if (SAVE_OUTPUTS && SAVE_CROPS) {
        fs::create_directories(crop_dir);
    }

    cv::Mat bgr = cv::imread(image_path);

    if (bgr.empty()) {
        throw std::runtime_error("Image not found: " + image_path);
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    std::vector<DetItem> det_items = detect(det_model, rgb, debug_bitmap_path);

    if (SAVE_OUTPUTS) {
        std::cout << "\nImage: " << image_path << std::endl;
        std::cout << "Detected boxes: " << det_items.size() << std::endl;
    }

    std::vector<OCRResult> results;

    for (size_t i = 0; i < det_items.size(); ++i) {
        const auto& item = det_items[i];
        cv::Mat crop_rgb = cropText(rgb, item.box);

        if (SAVE_OUTPUTS && SAVE_CROPS && !crop_rgb.empty()) {
            cv::Mat crop_bgr;
            cv::cvtColor(crop_rgb, crop_bgr, cv::COLOR_RGB2BGR);

            char name[256];
            std::snprintf(name, sizeof(name), "%s/crop_%03zu.png", crop_dir.c_str(), i + 1);
            cv::imwrite(name, crop_bgr);
        }

        auto [text, rec_score] = recognize(rec_model, crop_rgb, chars);

        if (SAVE_OUTPUTS && DEBUG_RAW_REC) {
            std::cout << std::setw(2) << std::setfill('0') << (i + 1)
                      << " RAW REC: " << text
                      << " | rec=" << rec_score
                      << " | det=" << item.det_score
                      << std::setfill(' ')
                      << std::endl;
        }

        if (!text.empty()) {
            OCRResult r;
            r.text = text;
            r.rec_score = rec_score;
            r.det_score = item.det_score;
            r.box = item.box;
            results.push_back(r);
        }
    }

    if (SAVE_OUTPUTS) {
        std::cout << "OCR RESULTS" << std::endl;
        std::cout << "================================================================================" << std::endl;

        for (const auto& r : results) {
            std::cout << r.text
                      << " | rec=" << r.rec_score
                      << " | det=" << r.det_score
                      << std::endl;
        }
    }

    return results;
}

// =========================================================
// MAIN
// =========================================================

int main() {
    try {
        if (SAVE_OUTPUTS) {
            fs::create_directories(OUTPUT_DIR);
            std::cout << "Loading ONNX Runtime models on CPU..." << std::endl;
        }

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "paddle_ocr_onnx_cpp");
        OnnxModel det_model(env, DET_MODEL);
        OnnxModel rec_model(env, REC_MODEL);
        std::vector<std::string> chars = loadDict(REC_DICT);

        if (SAVE_OUTPUTS) {
            std::cout << "Detection input: " << det_model.input_name << std::endl;
            std::cout << "Detection output: " << det_model.output_name << std::endl;
            std::cout << "Recognition input: " << rec_model.input_name << std::endl;
            std::cout << "Recognition output: " << rec_model.output_name << std::endl;
        }

        std::vector<std::string> image_paths = getImagePaths(TEST_IMAGE_DIR);
        if (image_paths.empty()) {
            throw std::runtime_error("No images found in: " + TEST_IMAGE_DIR);
        }

        std::vector<std::string> image_jsons;

        for (const auto& image_path : image_paths) {
            std::vector<OCRResult> results = processImage(image_path, det_model, rec_model, chars);
            std::string image_stem = getStem(image_path);

            if (SAVE_OUTPUTS) {
                std::string vis_path = OUTPUT_DIR + "/" + image_stem + "_onnx.png";
                std::string json_path = OUTPUT_DIR + "/" + image_stem + "_onnx.json";
                std::string crop_dir = OUTPUT_DIR + "/" + image_stem + "_onnx_debug_crops";

                saveJson(json_path, image_path, DET_MODEL, REC_MODEL, results);
                drawSideBySide(image_path, results, vis_path);

                std::cout << "Saved visualization: " << vis_path << std::endl;
                std::cout << "Saved JSON: " << json_path << std::endl;
                if (SAVE_CROPS) {
                    std::cout << "Saved crops: " << crop_dir << std::endl;
                }
            }

            image_jsons.push_back(buildImageJson(image_path, DET_MODEL, REC_MODEL, results, 4));
        }

        if (SAVE_OUTPUTS) {
            std::cout << "\nJSON OUTPUT" << std::endl;
        }

        std::cout << "{\n";
        std::cout << "  \"engine\": \"onnx\",\n";
        std::cout << "  \"image_dir\": \"" << jsonEscape(TEST_IMAGE_DIR) << "\",\n";
        std::cout << "  \"save_outputs\": " << (SAVE_OUTPUTS ? "true" : "false") << ",\n";
        std::cout << "  \"outputs\": [\n";

        for (size_t i = 0; i < image_jsons.size(); ++i) {
            std::cout << image_jsons[i];
            if (i + 1 < image_jsons.size()) {
                std::cout << ",";
            }
            std::cout << "\n";
        }

        std::cout << "  ]\n";
        std::cout << "}\n";

        if (SAVE_OUTPUTS) {
            std::cout << "\nDone." << std::endl;
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }
}
