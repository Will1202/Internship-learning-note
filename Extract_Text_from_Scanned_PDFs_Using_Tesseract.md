# Extract_Text_from_Scanned_PDFs_Using_Tesseract
OCR：教机器从图像中读取文本
到目前为止，您已经熟悉 OCR（光学字符识别）——将文本图像转换为实际的、可编辑的文本的过程。
我们将从 Tesseract 开始我们的旅程，Tesseract 是 Google 开发的开源 OCR 引擎，用于从扫描文档中提取文本。
## 第 1 步：设置您的环境
- PyMuPDF （fitz） – 读取 PDF 页面并将其转换为图像。
- Tesseract OCR – 从扫描图像中提取文本。
- OpenCV – 帮助清理图像并在检测到的文本周围绘制框。
- Pillow(PIL) – 让我们在 Colab 中显示和处理图像
- NumPy – 处理数字并使处理图像更容易。
⚠️重要！ 如果您使用的是 Google Colab，则必须手动安装 Tesseract OCR，因为默认情况下不包含它。

!apt install -y tesseract-ocr  # Install Tesseract OCR
!pip install pymupdf pytesseract opencv-python pillow numpy  # Install required Python libraries

