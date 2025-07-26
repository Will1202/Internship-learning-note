# Extract_Text_from_Scanned_PDFs_Using_Tesseract
OCR：教机器从图像中读取文本

到目前为止，您已经熟悉 OCR（光学字符识别）将文本图像转换为实际的、可编辑的文本的过程。

我们将从 Tesseract 开始我们的旅程，Tesseract 是 Google 开发的开源 OCR 引擎，用于从扫描文档中提取文本。
## 第 1 步：设置您的环境
- PyMuPDF （fitz） – 读取 PDF 页面并将其转换为图像。
- Tesseract OCR – 从扫描图像中提取文本。
- OpenCV – 帮助清理图像并在检测到的文本周围绘制框。
- Pillow(PIL) – 让我们在 Colab 中显示和处理图像
- NumPy – 处理数字并使处理图像更容易。

⚠️重要！ 如果您使用的是 Google Colab，则必须手动安装 Tesseract OCR，因为默认情况下不包含它。
```
!apt install -y tesseract-ocr  # Install Tesseract OCR
!pip install pymupdf pytesseract opencv-python pillow numpy  # Install required Python libraries
```
## 第 2 步：导入库
```
import cv2
import pytesseract
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
```
## 第 3 步：加载扫描的 PDF 并执行直接提取
让我们首先测试我们是否可以在没有 OCR 的情况下提取文本。如果 PDF 包含可搜索的文本，我们应该获得可读的内容。但是，如果 PDF 只是一张图像（如扫描的文档），这种方法可能会失败——这向我们展示了为什么需要 OCR。

使用 PyMuPDF 加载 PDF： 首先，让我们打开示例抵押贷款文档并尝试在没有 OCR 的情况下提取文本。
```
# Load the scanned PDF
pdf_path = "sample_mortgage_document.pdf"  # Update this path if needed
doc = fitz.open(pdf_path)

# Extract text from the first page using a standard method (this will fail for scanned PDFs)
page = doc[0]  # Get the first page
text = page.get_text("text")  # Try extracting text

print("Extracted Text (Without OCR):")
print(text)
```
## 第 4 步：将 PDF 页面转换为图像
由于文本提取失败，我们现在需要将 PDF 页面转换为图像，然后才能使用 OCR。在这里，我们正在转换 PDF 的第一页。
```
# Convert the first page to an image
pix = page.get_pixmap()
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# Display the image
display(img)
```
为什么要转换为图像？
- 扫描的 PDF 基本上是文本照片，而不是真实文本。
- OCR（光学字符识别）不能直接用于 PDF，它需要文本图像。
## 第 5 步：预处理图像以提高 OCR 准确性
**为什么我们需要预处理？**

当文本清晰且定义明确时，OCR 效果最佳。如果图像嘈杂、模糊或对比度差，OCR 引擎可能难以正确识别字符。
<img width="1077" height="748" alt="image" src="https://github.com/user-attachments/assets/b0ec4ab1-b7f7-4f68-a072-6a4a2a9d2ad2" />

### 将图像转换为灰度：灰度转换简化了 OCR 处理，使文本更易于检测。
首先，我们删除颜色信息，因为 OCR 只需要黑白文本。
```
# Convert the image to grayscale
img = np.array(img)  # Convert PIL image to NumPy array
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Display the grayscale image
display(Image.fromarray(gray))
```
**为什么选择灰度？**
- 减少不必要的细节。
- 帮助 OCR 专注于实际文本，而不是颜色或背景。
### 使用自适应阈值增强对比度
某些扫描的文档具有不均匀的光线或文本褪色。此步骤可提高文本可见性。为了使文本脱颖而出，我们应用了自适应阈值。
```
# Apply adaptive thresholding to enhance contrast
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Display the processed image
display(Image.fromarray(gray))
```
**为什么要增强对比度？**
- 使OCR的微弱文本更粗。
- 处理光照中的阴影和变化。
 
### 通过双边滤波降低噪声
OCR可能会将噪音误认为文本，从而导致错误。双边过滤可平滑背景，同时保持文本清晰。
```
# Apply Bilateral Filtering for noise reduction
gray = cv2.bilateralFilter(gray, 9, 75, 75)

# Display the processed image
display(Image.fromarray(gray))
```
**为什么要降噪？**
- 去除 OCR 可能误解的不需要的斑点。
- 保持文本边缘清晰，使字符不会模糊。
### 调整图像大小以提高OCR精度
如果扫描的文本太小，OCR可能会遗漏细节。我们放大图像以改进检测。
```
# Resize image for better OCR accuracy
scale_percent = 200  # Increase image size by 200%
width = int(gray.shape[1] * scale_percent / 100)
height = int(gray.shape[0] * scale_percent / 100)
gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

# Display the resized image
display(Image.fromarray(gray))
```
**为什么要调整大小？**
- 放大小文本，以便OCR可以更准确地识别字母。
- 防止OCR误读微小字符。

## 步骤 6：对预处理图像运行 OCR
使用 Tesseract OCR 提取文本，Tesseract 提供了多种文本提取选项，但对于扫描文档来说，最好的方法是使用优化的配置。
```
# Run OCR on the preprocessed image
custom_config = r'--oem 3 -l eng'
ocr_text = pytesseract.image_to_string(gray, config=custom_config)

# Print extracted text
print("OCR Extracted Text:\n")
print(ocr_text)
```
**-oem 3**: 使用基于AI（基于LSTM的模型）的最佳OCR引擎。
**l eng**:指定文本为英文。
如果一切正常，我们应该会看到从扫描文档中提取的文本打印在终端上。
然而，OCR 并非完美无缺——有时它会误读字符、添加空格或漏掉单词。

## 清理 OCR 输出
OCR 文本经常包含错误、不必要的空格或奇怪的格式。我们需要对其进行清理，使其更具可读性和结构性。
- Tesseract 有时会将文本拆分成多行，但实际上并不应该拆分。让我们来移除这些多余的换行符和空格。
```
# Remove excessive newlines and extra spaces
ocr_text = " ".join(ocr_text.split())
print("Cleaned OCR Text:\n", ocr_text)
```
**.split()** 将文本分解为单词列表。
**" ".join(... )**将单词用单个空格重新组合在一起，删除多余的空格和换行符。

 
