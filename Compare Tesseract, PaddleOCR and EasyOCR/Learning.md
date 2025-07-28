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

## 步骤 2：设置导入
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
```
ocr = PaddleOCR(use_textline_orientation=True, lang='en')
result = ocr.ocr(image_path)
```
PaddleOCR 读取文档后，会给出搜索结果列表。对于每个单词或短语，它会告诉您：
- 文本内容
- 对此有多大信心
- 文本在页面上的位置（以多边形框表示）

## 步骤 5：使用 OpenCV 手动绘制边界框
```
# Load image using OpenCV
img = cv2.imread(image_path)

# Draw boxes
for line in result[0]:
    box, text_info = line
    text, score = text_info
    box = [(int(pt[0]), int(pt[1])) for pt in box]
    cv2.polylines(img, [np.array(box)], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.putText(img, text, box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)    
```
<img width="1094" height="685" alt="image" src="https://github.com/user-attachments/assets/546fcfcf-897d-4e2c-b5da-d246ad695a50" />

## 步骤6：显示结果
```
# Convert and display the result
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)
plt.axis("off")
plt.show()
```
现在您应该会看到 PDF 页面，其中包含显示文本位置的方框和标签。即使是位置奇怪或旋转的文本也会被识别出来，因为这些方框是基于多边形的，而不是简单的矩形。

# 在 PDF 上运行 EasyOCR
## 步骤 1：安装 EasyOCR 和一些帮助程序
在开始之前，我们需要一些工具：
```
!pip install easyocr
!pip install pdf2image
!apt-get install poppler-utils -y
```
**pdf2image**将 PDF 页面转换为图像并**poppler-utils**帮助 PDF 渲染
## 第 2 步：上传并转换 PDF
我们将把PDF 的第一页转换为高质量图像（就像我们使用 Tesseract 和 PaddleOCR 所做的那样）。
```
from google.colab import files
from pdf2image import convert_from_path
from PIL import Image
import matplotlib.pyplot as plt

# Upload your file
uploaded = files.upload()
pdf_path = f"/content/{list(uploaded.keys())[0]}"

# Convert first page to image
images = convert_from_path(pdf_path, dpi=300)
image = images[0]
image_path = '/content/page.png'
image.save(image_path)

# Show preview
plt.imshow(image)
plt.axis('off')
plt.title("Uploaded PDF - Page 1")
plt.show()
```
功能：将您的 PDF 转换为 EasyOCR 可以读取的 PNG 图像。为了简单起见，我们暂时只使用第一页。

## 步骤 3：运行 EasyOCR 并可视化结果
现在是时候提取文本并绘制框，以便我们可以看到 EasyOCR 拾取了什么。
```
import easyocr
from PIL import ImageDraw

# Initialize reader
reader = easyocr.Reader(['en'])

# Run OCR
result = reader.readtext(image_path)
```
```
# Visualize
img_copy = image.copy()
draw = ImageDraw.Draw(img_copy)
extracted_text = []

for (bbox, text, confidence) in result:
    if confidence > 0.5:
        # Bounding box
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        draw.rectangle([top_left, bottom_right], outline='red', width=2)

        # Confidence score
        draw.text((top_left[0], top_left[1] - 20), f"{confidence:.2f}", fill='red')

        extracted_text.append(text)

plt.figure(figsize=(15, 10))
plt.imshow(img_copy)
plt.axis('off')
plt.title(f"EasyOCR Results – {len(extracted_text)} segments")
plt.show()
```
## 步骤4: 打印提取的文本
现在让我们以更清晰的格式查看原始结果：
```
print(f"\n📝 Extracted Text ({len(extracted_text)} segments):")
for i, text in enumerate(extracted_text, 1):
    print(f"{i:2d}. {text}")
```


# 三种OCR的比较
<img width="890" height="193" alt="image" src="https://github.com/user-attachments/assets/a86ebcd7-ac52-44e0-b9b2-082498455019" />

 




