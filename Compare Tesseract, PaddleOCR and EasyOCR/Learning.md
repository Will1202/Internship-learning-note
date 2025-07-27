- **PaddleOCR**：由百度打造的高性能 OCR 引擎，能够出色地处理密集复杂的文档。它内置数十个预训练模型，并支持多语言识别。设置过程略长，但一旦启动，即可提供强大且结构化的输出，非常适合包含段落、字段和小文本的文档。
- **EasyOCR**：顾名思义，这款工具非常易于使用。它轻巧易用，安装快捷，非常适合扫描表格和简单的布局，例如抵押贷款工作表、身份证或税表。它可能无法像 PaddleOCR 那样捕捉深层结构布局，但它可以快速提供良好的结果，并能很好地处理大多数基于字段的表单。
<img width="1089" height="501" alt="image" src="https://github.com/user-attachments/assets/ba1b2dbc-f46f-4379-b7c3-d91e171f4567" />

# PaddleOCR 实际操作
## 步骤 1：安装库
```
!pip install paddleocr
!pip install paddlepaddle
!apt install poppler-utils
```
- PaddlePaddle是 PaddleOCR 运行的深度学习框架（就像该工具背后的引擎一样）。
- Poppler是一款帮助将 PDF 文件转换为图像的实用程序。

## 第 2 步：设置导入
```
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import cv2
import matplotlib.pyplot as plt
from PIL import Image
 ```
## 步骤 3：将 PDF 转换为图像
与 Tesseract 一样，PaddleOCR 也适用于图像，因此让我们将 PDF 的第一页转换为干净的图像。
```
# Convert first page to image
images = convert_from_path("/content/LenderFeesWorksheetNew.pdf", dpi=300)
image_path = "page_1.png"
images[0].save(image_path, "PNG")
```
 
## 步骤4：在图像上运行PaddleOCR
ocr = PaddleOCR(use_textline_orientation=True, lang='en')
result = ocr.ocr(image_path)
```
PaddleOCR 读取文档后，会给出搜索结果列表。对于每个单词或短语，它会告诉您：
- 文本内容
- 对此有多大信心
- 文本在页面上的位置（以多边形框表示）



