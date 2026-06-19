// Required NuGet packages:
//   Microsoft.ML.OnnxRuntime
//   OpenCvSharp4
//   OpenCvSharp4.runtime.linux-x64

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

public class OnnxInference
{
    // =========================================================
    // CONFIG
    // =========================================================

    // Detection model path. This ONNX model finds text boxes from the full image.
    static readonly string DET_MODEL = "models_onnx/PP-OCRv5_server_det_infer.onnx";

    // Recognition model path. This ONNX model reads text from each cropped text box.
    static readonly string REC_MODEL = "models_onnx/korean_PP-OCRv5_mobile_rec.onnx";

    // PaddleOCR dictionary path. Index 0 is CTC blank; real characters start from index 1.
    static readonly string REC_DICT = "korean_dict.txt";

    // Test image directory. Every supported image in this directory will be processed.
    static readonly string TEST_IMAGE_DIR = "./data";

    // Output directory for JSON, visualization images, debug bitmap, and crops.
    static readonly string OUTPUT_DIR = "./recognition_results";

    // Set true to save JSON/images/crops. Set false to print JSON only in terminal.
    static readonly bool SAVE_OUTPUTS = true;

    // Save cropped text line images for debugging detector and recognizer alignment.
    static readonly bool SAVE_CROPS = true;

    // Print raw recognizer output for every detected crop while SAVE_OUTPUTS is true.
    static readonly bool DEBUG_RAW_REC = true;

    // PaddleOCR DB detector postprocess values.
    static readonly float DET_THRESH = 0.30f;
    static readonly float BOX_THRESH = 0.30f;
    static readonly float UNCLIP_RATIO = 1.7f;
    static readonly int DET_LIMIT_SIDE_LEN = 1536;

    // PaddleOCR recognition resize values.
    static readonly int DEFAULT_REC_HEIGHT = 48;
    static readonly int MAX_DYNAMIC_REC_WIDTH = 2048;

    static readonly string[] IMAGE_EXTENSIONS =
    {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    };

    // =========================================================
    // STRUCTS
    // =========================================================

    class DetItem
    {
        public Point2f[] Box = Array.Empty<Point2f>();
        public float DetScore;
    }

    class OCRResult
    {
        public string Text = "";
        public float RecScore;
        public float DetScore;
        public Point2f[] Box = Array.Empty<Point2f>();
    }

    class ImageOutput
    {
        public string image { get; set; } = "";
        public string det_model { get; set; } = "";
        public string rec_model { get; set; } = "";
        public List<object> results { get; set; } = new();
    }

    // =========================================================
    // UTILS
    // =========================================================

    static List<string> GetImagePaths(string imageDir)
    {
        if (File.Exists(imageDir) && IMAGE_EXTENSIONS.Contains(Path.GetExtension(imageDir).ToLowerInvariant()))
        {
            return new List<string> { imageDir };
        }

        if (!Directory.Exists(imageDir))
        {
            throw new FileNotFoundException("Image path not found: " + imageDir);
        }

        return Directory.GetFiles(imageDir)
            .Where(p => IMAGE_EXTENSIONS.Contains(Path.GetExtension(p).ToLowerInvariant()))
            .OrderBy(p => p)
            .ToList();
    }

    static bool ContainsKorean(string text)
    {
        // Hangul syllables live in Unicode range U+AC00 - U+D7A3.
        foreach (char ch in text)
        {
            if (ch >= '\uAC00' && ch <= '\uD7A3')
            {
                return true;
            }
        }

        return false;
    }

    static Rect SafeRect(int x1, int y1, int x2, int y2, int width, int height)
    {
        x1 = Math.Clamp(x1, 0, width - 1);
        y1 = Math.Clamp(y1, 0, height - 1);
        x2 = Math.Clamp(x2, 0, width - 1);
        y2 = Math.Clamp(y2, 0, height - 1);

        if (x2 < x1) (x1, x2) = (x2, x1);
        if (y2 < y1) (y1, y2) = (y2, y1);

        return new Rect(x1, y1, Math.Max(1, x2 - x1 + 1), Math.Max(1, y2 - y1 + 1));
    }

    // =========================================================
    // DICTIONARY
    // =========================================================

    static List<string> LoadDict(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("Dictionary not found: " + path);
        }

        // PaddleOCR CTC uses index 0 as blank, then dictionary characters.
        var chars = new List<string> { "blank" };

        foreach (string line in File.ReadLines(path, Encoding.UTF8))
        {
            string item = line.TrimEnd('\r', '\n');
            if (item.Length > 0)
            {
                chars.Add(item);
            }
        }

        // PaddleOCR recognition dictionaries usually keep space as the last class.
        if (chars.Count == 0 || chars[^1] != " ")
        {
            chars.Add(" ");
        }

        if (SAVE_OUTPUTS)
        {
            Console.WriteLine($"Dictionary classes: {chars.Count}");
        }

        return chars;
    }

    // =========================================================
    // PREPROCESSING
    // =========================================================

    static Mat ResizeDet(Mat rgb, int limitSideLen = 1536)
    {
        int h = rgb.Rows;
        int w = rgb.Cols;

        // PaddleOCR keeps detector dimensions divisible by 32.
        float ratio = Math.Min(limitSideLen / (float)Math.Max(h, w), 1.0f);
        int resizeH = (int)Math.Round(h * ratio / 32.0f) * 32;
        int resizeW = (int)Math.Round(w * ratio / 32.0f) * 32;
        resizeH = Math.Max(32, resizeH);
        resizeW = Math.Max(32, resizeW);

        var resized = new Mat();
        Cv2.Resize(rgb, resized, new Size(resizeW, resizeH));
        return resized;
    }

    static DenseTensor<float> MatToDetTensor(Mat rgb)
    {
        int h = rgb.Rows;
        int w = rgb.Cols;
        var tensor = new DenseTensor<float>(new[] { 1, 3, h, w });

        float[] mean = { 0.485f, 0.456f, 0.406f };
        float[] std = { 0.229f, 0.224f, 0.225f };

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                Vec3b p = rgb.At<Vec3b>(y, x);
                for (int c = 0; c < 3; c++)
                {
                    float v = p[c] / 255.0f;
                    tensor[0, c, y, x] = (v - mean[c]) / std[c];
                }
            }
        }

        return tensor;
    }

    static DenseTensor<float> MatToRecTensor(Mat rgb)
    {
        int h = rgb.Rows;
        int w = rgb.Cols;
        var tensor = new DenseTensor<float>(new[] { 1, 3, h, w });

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                Vec3b p = rgb.At<Vec3b>(y, x);
                for (int c = 0; c < 3; c++)
                {
                    float v = p[c] / 255.0f;
                    tensor[0, c, y, x] = (v - 0.5f) / 0.5f;
                }
            }
        }

        return tensor;
    }

    // =========================================================
    // DETECTION POSTPROCESSING
    // =========================================================

    static Point2f[] OrderPoints(IEnumerable<Point2f> input)
    {
        var pts = input.ToArray();
        var rect = new Point2f[4];

        rect[0] = pts.OrderBy(p => p.X + p.Y).First();
        rect[2] = pts.OrderByDescending(p => p.X + p.Y).First();
        rect[1] = pts.OrderBy(p => p.Y - p.X).First();
        rect[3] = pts.OrderByDescending(p => p.Y - p.X).First();

        return rect;
    }

    static (Point2f[] Box, float ShortSide) GetMiniBox(Point[] contour)
    {
        RotatedRect rr = Cv2.MinAreaRect(contour);
        Point2f[] box = OrderPoints(rr.Points());

        float side1 = (float)Math.Sqrt(Math.Pow(box[0].X - box[1].X, 2) + Math.Pow(box[0].Y - box[1].Y, 2));
        float side2 = (float)Math.Sqrt(Math.Pow(box[1].X - box[2].X, 2) + Math.Pow(box[1].Y - box[2].Y, 2));

        return (box, Math.Min(side1, side2));
    }

    static (Point2f[] Box, float ShortSide) GetMiniBoxF(Point2f[] contour)
    {
        Point[] points = contour
            .Select(p => new Point((int)Math.Round(p.X), (int)Math.Round(p.Y)))
            .ToArray();

        return GetMiniBox(points);
    }

    static float BoxScore(Mat prob, Point2f[] box)
    {
        int h = prob.Rows;
        int w = prob.Cols;

        int xmin = Math.Max(0, (int)Math.Floor(box.Min(p => p.X)));
        int xmax = Math.Min(w - 1, (int)Math.Ceiling(box.Max(p => p.X)));
        int ymin = Math.Max(0, (int)Math.Floor(box.Min(p => p.Y)));
        int ymax = Math.Min(h - 1, (int)Math.Ceiling(box.Max(p => p.Y)));

        if (xmax <= xmin || ymax <= ymin)
        {
            return 0.0f;
        }

        var mask = Mat.Zeros(ymax - ymin + 1, xmax - xmin + 1, MatType.CV_8UC1);
        Point[] localBox = box
            .Select(p => new Point((int)Math.Round(p.X - xmin), (int)Math.Round(p.Y - ymin)))
            .ToArray();

        Cv2.FillPoly(mask, new[] { localBox }, Scalar.White);

        using Mat roi = prob[new Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)];
        return (float)Cv2.Mean(roi, mask)[0];
    }

    static Point2f[] UnclipApprox(Point2f[] box, float ratio = 1.7f)
    {
        // This expands around the center. For exact PaddleOCR parity, replace with Clipper2 offset.
        float cx = box.Average(p => p.X);
        float cy = box.Average(p => p.Y);

        return box.Select(p => new Point2f(
            cx + (p.X - cx) * ratio,
            cy + (p.Y - cy) * ratio
        )).ToArray();
    }

    static List<DetItem> Detect(InferenceSession detSession, Mat rgb, string? debugBitmapPath)
    {
        using Mat resized = ResizeDet(rgb, DET_LIMIT_SIDE_LEN);
        var inputTensor = MatToDetTensor(resized);
        string inputName = detSession.InputMetadata.Keys.First();
        string outputName = detSession.OutputMetadata.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = detSession.Run(inputs);
        Tensor<float> pred = outputs.First(v => v.Name == outputName).AsTensor<float>();
        int probH = pred.Dimensions[2];
        int probW = pred.Dimensions[3];

        var prob = new Mat(probH, probW, MatType.CV_32FC1);
        for (int y = 0; y < probH; y++)
        {
            for (int x = 0; x < probW; x++)
            {
                prob.Set(y, x, pred[0, 0, y, x]);
            }
        }

        var bitmap = new Mat();
        Cv2.Threshold(prob, bitmap, DET_THRESH, 255, ThresholdTypes.Binary);
        bitmap.ConvertTo(bitmap, MatType.CV_8UC1);

        if (!string.IsNullOrEmpty(debugBitmapPath))
        {
            Cv2.ImWrite(debugBitmapPath, bitmap);
        }

        Cv2.FindContours(bitmap, out Point[][] contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

        int srcH = rgb.Rows;
        int srcW = rgb.Cols;
        var boxes = new List<DetItem>();

        foreach (Point[] contour in contours)
        {
            if (contour.Length < 3)
            {
                continue;
            }

            var (box, shortSide) = GetMiniBox(contour);
            if (shortSide < 3.0f)
            {
                continue;
            }

            float score = BoxScore(prob, box);
            if (score < BOX_THRESH)
            {
                continue;
            }

            Point2f[] expanded = UnclipApprox(box, UNCLIP_RATIO);
            var (box2, shortSide2) = GetMiniBoxF(expanded);
            if (shortSide2 < 3.0f)
            {
                continue;
            }

            for (int i = 0; i < box2.Length; i++)
            {
                box2[i].X = Math.Clamp((float)Math.Round(box2[i].X / probW * srcW), 0, srcW - 1);
                box2[i].Y = Math.Clamp((float)Math.Round(box2[i].Y / probH * srcH), 0, srcH - 1);
            }

            boxes.Add(new DetItem { Box = OrderPoints(box2), DetScore = score });
        }

        return boxes
            .OrderBy(b => b.Box.Min(p => p.Y))
            .ThenBy(b => b.Box.Min(p => p.X))
            .ToList();
    }

    // =========================================================
    // CROP
    // =========================================================

    static Mat CropText(Mat rgb, Point2f[] inputBox)
    {
        Point2f[] box = OrderPoints(inputBox);

        int cropW = (int)Math.Max(
            Math.Sqrt(Math.Pow(box[0].X - box[1].X, 2) + Math.Pow(box[0].Y - box[1].Y, 2)),
            Math.Sqrt(Math.Pow(box[2].X - box[3].X, 2) + Math.Pow(box[2].Y - box[3].Y, 2))
        );

        int cropH = (int)Math.Max(
            Math.Sqrt(Math.Pow(box[0].X - box[3].X, 2) + Math.Pow(box[0].Y - box[3].Y, 2)),
            Math.Sqrt(Math.Pow(box[1].X - box[2].X, 2) + Math.Pow(box[1].Y - box[2].Y, 2))
        );

        cropW = Math.Max(1, cropW);
        cropH = Math.Max(1, cropH);

        Point2f[] dst =
        {
            new(0, 0),
            new(cropW, 0),
            new(cropW, cropH),
            new(0, cropH)
        };

        using Mat matrix = Cv2.GetPerspectiveTransform(box, dst);
        var crop = new Mat();
        Cv2.WarpPerspective(rgb, crop, matrix, new Size(cropW, cropH), InterpolationFlags.Cubic, BorderTypes.Replicate);

        if (crop.Rows / (float)Math.Max(1, crop.Cols) >= 1.5f)
        {
            var rotated = new Mat();
            Cv2.Rotate(crop, rotated, RotateFlags.Rotate90Counterclockwise);
            crop.Dispose();
            crop = rotated;
        }

        return crop;
    }

    // =========================================================
    // RECOGNITION
    // =========================================================

    static int GetRecHeight(InferenceSession recSession)
    {
        string inputName = recSession.InputMetadata.Keys.First();
        int[] dims = recSession.InputMetadata[inputName].Dimensions;

        if (dims.Length >= 4 && dims[2] > 0)
        {
            return dims[2];
        }

        return DEFAULT_REC_HEIGHT;
    }

    static Mat ResizeRec(Mat crop, int recH, int maxDynamicW)
    {
        int h = crop.Rows;
        int w = crop.Cols;

        if (h <= 0 || w <= 0)
        {
            return new Mat();
        }

        float ratio = w / (float)h;
        int newW = (int)Math.Ceiling(recH * ratio);
        int finalW = Math.Min(maxDynamicW, Math.Max(32, newW));

        newW = Math.Min(newW, finalW);
        newW = Math.Max(16, newW);

        var resized = new Mat();
        Cv2.Resize(crop, resized, new Size(newW, recH), 0, 0, InterpolationFlags.Linear);

        var padded = new Mat(recH, finalW, MatType.CV_8UC3, new Scalar(255, 255, 255));
        resized.CopyTo(padded[new Rect(0, 0, newW, recH)]);
        resized.Dispose();

        return padded;
    }

    static (string Text, float Score) DecodeCTC(Tensor<float> pred, List<string> chars)
    {
        int rank = pred.Dimensions.Length;
        int t;
        int c;
        bool transposed = false;

        if (rank == 3)
        {
            if (pred.Dimensions[1] == chars.Count)
            {
                c = pred.Dimensions[1];
                t = pred.Dimensions[2];
                transposed = true;
            }
            else
            {
                t = pred.Dimensions[1];
                c = pred.Dimensions[2];
            }
        }
        else if (rank == 2)
        {
            if (pred.Dimensions[0] == chars.Count)
            {
                c = pred.Dimensions[0];
                t = pred.Dimensions[1];
                transposed = true;
            }
            else
            {
                t = pred.Dimensions[0];
                c = pred.Dimensions[1];
            }
        }
        else
        {
            throw new Exception("Recognition output shape must be 2D or 3D.");
        }

        if (SAVE_OUTPUTS && c != chars.Count)
        {
            Console.WriteLine($"WARNING: dict classes={chars.Count}, model classes={c}");
        }

        var text = new StringBuilder();
        var scores = new List<float>();
        int last = -1;

        for (int i = 0; i < t; i++)
        {
            int bestIdx = 0;
            float bestVal = rank == 3
                ? (transposed ? pred[0, 0, i] : pred[0, i, 0])
                : (transposed ? pred[0, i] : pred[i, 0]);

            for (int j = 1; j < c; j++)
            {
                float value = rank == 3
                    ? (transposed ? pred[0, j, i] : pred[0, i, j])
                    : (transposed ? pred[j, i] : pred[i, j]);
                if (value > bestVal)
                {
                    bestVal = value;
                    bestIdx = j;
                }
            }

            // CTC removes blank and repeated tokens.
            if (bestIdx != 0 && bestIdx != last && bestIdx < chars.Count)
            {
                text.Append(chars[bestIdx]);
                scores.Add(bestVal);
            }

            last = bestIdx;
        }

        float score = scores.Count > 0 ? scores.Average() : 0.0f;
        return (text.ToString(), score);
    }

    static (string Text, float Score) Recognize(InferenceSession recSession, Mat crop, List<string> chars)
    {
        int recH = GetRecHeight(recSession);
        using Mat recImg = ResizeRec(crop, recH, MAX_DYNAMIC_REC_WIDTH);

        if (recImg.Empty())
        {
            return ("", 0.0f);
        }

        var inputTensor = MatToRecTensor(recImg);
        string inputName = recSession.InputMetadata.Keys.First();
        string outputName = recSession.OutputMetadata.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = recSession.Run(inputs);
        Tensor<float> pred = outputs.First(v => v.Name == outputName).AsTensor<float>();

        return DecodeCTC(pred, chars);
    }

    // =========================================================
    // JSON AND VISUALIZATION
    // =========================================================

    static object ToJsonResult(OCRResult r)
    {
        return new
        {
            text = r.Text,
            rec_score = r.RecScore,
            det_score = r.DetScore,
            box = r.Box.Select(p => new[] { (int)Math.Round(p.X), (int)Math.Round(p.Y) }).ToArray()
        };
    }

    static Rect PolygonBBox(Point2f[] box, int imgW, int imgH)
    {
        return SafeRect(
            (int)Math.Floor(box.Min(p => p.X)),
            (int)Math.Floor(box.Min(p => p.Y)),
            (int)Math.Ceiling(box.Max(p => p.X)),
            (int)Math.Ceiling(box.Max(p => p.Y)),
            imgW,
            imgH
        );
    }

    static Scalar[] PaletteBgr()
    {
        return new[]
        {
            new Scalar(121, 236, 202),
            new Scalar(196, 230, 125),
            new Scalar(226, 161, 185),
            new Scalar(255, 203, 134),
            new Scalar(102, 180, 246),
            new Scalar(190, 245, 152),
            new Scalar(230, 225, 210),
            new Scalar(120, 225, 255)
        };
    }

    static void BlendRect(Mat img, Rect rect, Scalar color, double alpha)
    {
        Rect r = rect.Intersect(new Rect(0, 0, img.Cols, img.Rows));
        if (r.Width <= 0 || r.Height <= 0)
        {
            return;
        }

        using Mat roi = img[r];
        using Mat colorMat = new Mat(roi.Size(), roi.Type(), color);
        Cv2.AddWeighted(colorMat, alpha, roi, 1.0 - alpha, 0.0, roi);
        Cv2.Rectangle(img, r, color, 1);
    }

    static void DrawSideBySide(string imagePath, List<OCRResult> results, string savePath)
    {
        using Mat leftBgr = Cv2.ImRead(imagePath);
        if (leftBgr.Empty())
        {
            throw new FileNotFoundException("Image not found: " + imagePath);
        }

        int imgW = leftBgr.Cols;
        int imgH = leftBgr.Rows;
        int gap = 30;
        int rx = imgW + gap;

        using Mat canvas = new Mat(imgH, imgW + gap + imgW, MatType.CV_8UC3, new Scalar(255, 255, 255));
        leftBgr.CopyTo(canvas[new Rect(0, 0, imgW, imgH)]);
        Cv2.Line(canvas, new Point(imgW + gap / 2, 0), new Point(imgW + gap / 2, imgH), new Scalar(220, 220, 220), 1);

        Scalar[] colors = PaletteBgr();

        for (int i = 0; i < results.Count; i++)
        {
            OCRResult r = results[i];
            Scalar color = colors[i % colors.Length];
            Rect bbox = PolygonBBox(r.Box, imgW, imgH);

            Rect leftRect = SafeRect(
                bbox.X - 2,
                bbox.Y - 2,
                bbox.X + bbox.Width + 2,
                bbox.Y + bbox.Height + 2,
                imgW,
                imgH
            );

            BlendRect(canvas, leftRect, color, 0.42);

            int rightX1 = rx + bbox.X;
            int rightY1 = Math.Max(0, bbox.Y - 2);
            int rightBoxW = Math.Max(1, bbox.Width);
            int rightBoxH = Math.Max(1, bbox.Height);
            bool useCropForText = ContainsKorean(r.Text);

            Rect rightRect;
            if (useCropForText)
            {
                rightRect = SafeRect(rightX1, rightY1, rightX1 + rightBoxW + 8, rightY1 + rightBoxH + 5, canvas.Cols, canvas.Rows);
            }
            else
            {
                Size textSize = Cv2.GetTextSize(r.Text, HersheyFonts.HersheySimplex, 0.55, 1, out _);
                rightRect = SafeRect(rightX1, rightY1, rightX1 + textSize.Width + 8, rightY1 + Math.Max(rightBoxH, textSize.Height + 6), canvas.Cols, canvas.Rows);
            }

            BlendRect(canvas, rightRect, color, 0.28);

            if (useCropForText)
            {
                Rect cropRect = bbox.Intersect(new Rect(0, 0, imgW, imgH));
                if (cropRect.Width > 0 && cropRect.Height > 0)
                {
                    using Mat crop = leftBgr[cropRect].Clone();
                    int targetW = Math.Max(1, rightRect.Width - 6);
                    int targetH = Math.Max(1, rightRect.Height - 4);
                    Cv2.Resize(crop, crop, new Size(targetW, targetH), 0, 0, InterpolationFlags.Lanczos4);

                    Rect pasteRect = new Rect(
                        rightRect.X + 3,
                        rightRect.Y + 2,
                        Math.Min(targetW, canvas.Cols - rightRect.X - 3),
                        Math.Min(targetH, canvas.Rows - rightRect.Y - 2)
                    );

                    if (pasteRect.Width > 0 && pasteRect.Height > 0)
                    {
                        crop[new Rect(0, 0, pasteRect.Width, pasteRect.Height)].CopyTo(canvas[pasteRect]);
                    }
                }
            }
            else
            {
                Cv2.PutText(canvas, r.Text, new Point(rightRect.X + 4, rightRect.Y + rightRect.Height - 5),
                    HersheyFonts.HersheySimplex, 0.55, new Scalar(0, 0, 0), 1, LineTypes.AntiAlias);
            }
        }

        Cv2.ImWrite(savePath, canvas);
    }

    // =========================================================
    // IMAGE PIPELINE
    // =========================================================

    static ImageOutput ProcessImage(string imagePath, InferenceSession detSession, InferenceSession recSession, List<string> chars)
    {
        string stem = Path.GetFileNameWithoutExtension(imagePath);
        string cropDir = Path.Combine(OUTPUT_DIR, $"{stem}_onnx_debug_crops");
        string bitmapPath = SAVE_OUTPUTS ? Path.Combine(OUTPUT_DIR, $"{stem}_onnx_debug_bitmap.png") : "";

        if (SAVE_OUTPUTS && SAVE_CROPS)
        {
            Directory.CreateDirectory(cropDir);
        }

        using Mat bgr = Cv2.ImRead(imagePath);
        if (bgr.Empty())
        {
            throw new FileNotFoundException("Image not found: " + imagePath);
        }

        using Mat rgb = new Mat();
        Cv2.CvtColor(bgr, rgb, ColorConversionCodes.BGR2RGB);

        List<DetItem> detItems = Detect(detSession, rgb, bitmapPath);

        if (SAVE_OUTPUTS)
        {
            Console.WriteLine($"\nImage: {imagePath}");
            Console.WriteLine($"Detected boxes: {detItems.Count}");
        }

        var results = new List<OCRResult>();

        for (int i = 0; i < detItems.Count; i++)
        {
            DetItem item = detItems[i];
            using Mat cropRgb = CropText(rgb, item.Box);

            if (SAVE_OUTPUTS && SAVE_CROPS && !cropRgb.Empty())
            {
                using Mat cropBgr = new Mat();
                Cv2.CvtColor(cropRgb, cropBgr, ColorConversionCodes.RGB2BGR);
                Cv2.ImWrite(Path.Combine(cropDir, $"crop_{i + 1:000}.png"), cropBgr);
            }

            var (text, recScore) = Recognize(recSession, cropRgb, chars);

            if (SAVE_OUTPUTS && DEBUG_RAW_REC)
            {
                Console.WriteLine($"{i + 1:00} RAW REC: {text} | rec={recScore:F4} | det={item.DetScore:F4}");
            }

            if (!string.IsNullOrWhiteSpace(text))
            {
                results.Add(new OCRResult
                {
                    Text = text,
                    RecScore = recScore,
                    DetScore = item.DetScore,
                    Box = item.Box
                });
            }
        }

        if (SAVE_OUTPUTS)
        {
            Console.WriteLine("OCR RESULTS");
            Console.WriteLine("================================================================================");
            foreach (OCRResult r in results)
            {
                Console.WriteLine($"{r.Text} | rec={r.RecScore:F4} | det={r.DetScore:F4}");
            }
        }

        var output = new ImageOutput
        {
            image = imagePath,
            det_model = DET_MODEL,
            rec_model = REC_MODEL,
            results = results.Select(ToJsonResult).ToList()
        };

        if (SAVE_OUTPUTS)
        {
            string visPath = Path.Combine(OUTPUT_DIR, $"{stem}_onnx.png");
            string jsonPath = Path.Combine(OUTPUT_DIR, $"{stem}_onnx.json");

            File.WriteAllText(jsonPath, JsonSerializer.Serialize(output, JsonOptions()), Encoding.UTF8);
            DrawSideBySide(imagePath, results, visPath);

            Console.WriteLine($"Saved visualization: {visPath}");
            Console.WriteLine($"Saved JSON: {jsonPath}");
            if (SAVE_CROPS)
            {
                Console.WriteLine($"Saved crops: {cropDir}");
            }
        }

        return output;
    }

    static JsonSerializerOptions JsonOptions()
    {
        return new JsonSerializerOptions
        {
            WriteIndented = true,
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
        };
    }

    // =========================================================
    // MAIN
    // =========================================================

    public static void Main()
    {
        if (SAVE_OUTPUTS)
        {
            Directory.CreateDirectory(OUTPUT_DIR);
            Console.WriteLine("Loading ONNX Runtime models on CPU...");
        }

        using var detSession = new InferenceSession(DET_MODEL);
        using var recSession = new InferenceSession(REC_MODEL);
        List<string> chars = LoadDict(REC_DICT);

        if (SAVE_OUTPUTS)
        {
            Console.WriteLine($"Detection input: {detSession.InputMetadata.Keys.First()}");
            Console.WriteLine($"Detection output: {detSession.OutputMetadata.Keys.First()}");
            Console.WriteLine($"Recognition input: {recSession.InputMetadata.Keys.First()}");
            Console.WriteLine($"Recognition output: {recSession.OutputMetadata.Keys.First()}");
        }

        List<string> imagePaths = GetImagePaths(TEST_IMAGE_DIR);
        if (imagePaths.Count == 0)
        {
            throw new Exception("No images found in: " + TEST_IMAGE_DIR);
        }

        var outputs = imagePaths.Select(p => ProcessImage(p, detSession, recSession, chars)).ToList();
        var terminalJson = new
        {
            engine = "onnx",
            image_dir = TEST_IMAGE_DIR,
            save_outputs = SAVE_OUTPUTS,
            outputs
        };

        if (SAVE_OUTPUTS)
        {
            Console.WriteLine("\nJSON OUTPUT");
        }

        Console.WriteLine(JsonSerializer.Serialize(terminalJson, JsonOptions()));

        if (SAVE_OUTPUTS)
        {
            Console.WriteLine("\nDone.");
        }
    }
}
