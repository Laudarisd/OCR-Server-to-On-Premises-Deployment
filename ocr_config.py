from pathlib import Path

# =========================================================
# COMMON CONFIG FOR OPENVINO AND ONNX OCR DEMOS
# =========================================================

# Choose a folder; every image in this folder will be processed.
TEST_IMAGE_DIR = Path("./data")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Set True to save visualization, JSON, debug bitmap, and crops.
# Set None to skip all file saving and only print JSON in the terminal.
SAVE_OUTPUTS = True
SAVE_DIR = Path("./recognition_results")

# Recognition dictionary exported from PaddleOCR.
REC_DICT = "korean_dict.txt"

# PaddleOCR DB detector postprocess values.
DET_THRESH = 0.30
BOX_THRESH = 0.30
UNCLIP_RATIO = 1.7
DET_LIMIT_SIDE_LEN = 1536

# PaddleOCR recognition resize values.
DEFAULT_REC_HEIGHT = 48
MAX_DYNAMIC_REC_WIDTH = 2048

# Debug text in terminal while each crop is recognized.
DEBUG_RAW_REC = True

# Save each recognized crop only when SAVE_OUTPUTS is True.
SAVE_CROPS = True


# =========================================================
# ONNX CONFIG
# =========================================================

ONNX_DET_MODEL = "models_onnx/PP-OCRv5_server_det_infer.onnx"
ONNX_REC_MODEL = "models_onnx/korean_PP-OCRv5_mobile_rec.onnx"


# =========================================================
# OPENVINO CONFIG
# =========================================================

OPENVINO_DET_MODEL = "models_openvino/PP-OCRv5_server_det_infer.xml"
OPENVINO_REC_MODEL = "models_openvino/korean_PP-OCRv5_mobile_rec.xml"
OPENVINO_DEVICE = "CPU"
