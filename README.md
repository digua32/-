# Google Colab 使用教學：附帶 MNIST 案例
11228017盧思綺 11224120丁誠祐
## 一、專案簡介

本專案旨在介紹 Google Colaboratory（Google Colab）的基本操作，並透過 MNIST 手寫數字辨識案例，示範如何使用 Python 與 TensorFlow 建立、訓練及評估人工智慧模型。

Google Colab 是一種雲端 Notebook 執行環境，使用者無須在本機安裝完整的 Python 開發環境，即可透過瀏覽器撰寫及執行程式碼。

本專案將從 Google Colab 的基本操作開始，逐步完成 MNIST 手寫數字辨識模型，協助初學者了解雲端程式開發與機器學習的基本流程。

## 二、專案目標

- 了解 Google Colab 的基本功能與使用方式。
- 學習如何建立、編輯及執行 Python 程式碼。
- 認識 MNIST 手寫數字資料集。
- 使用 TensorFlow 建立簡單的神經網路模型。
- 學習模型訓練、測試與準確率評估。
- 了解人工智慧影像分類的基本概念。

## 三、使用工具與技術

| 工具／技術 | 用途 |
| --- | --- |
| Google Colab | 雲端程式開發與執行環境 |
| Python | 主要程式設計語言 |
| TensorFlow / Keras | 建立與訓練機器學習模型 |
| MNIST | 手寫數字影像資料集 |
| Matplotlib | 顯示手寫數字影像與分析結果 |
| NumPy | 數值運算與資料處理 |

## 四、Google Colab 基本操作

### 4.1 開啟 Google Colab

1. 前往 [Google Colab](https://colab.research.google.com/)。
2. 使用 Google 帳號登入。
3. 點選「新增筆記本」建立新的 Notebook。
4. 在程式碼儲存格中輸入 Python 程式碼。
5. 點選執行按鈕，或使用 `Shift + Enter` 執行程式。

### 4.2 執行第一個 Python 程式

```python
print("Hello, Google Colab!")
```

執行結果：

```text
Hello, Google Colab!
```

### 4.3 Notebook 基本概念

Google Colab 的 Notebook 主要由不同的儲存格組成：

- **程式碼儲存格（Code Cell）：** 撰寫與執行 Python 程式。
- **文字儲存格（Text Cell）：** 使用 Markdown 撰寫說明文件。
- **執行結果（Output）：** 顯示程式執行後的輸出內容。

## 五、MNIST 手寫數字辨識案例

### 5.1 案例介紹

MNIST 是常見的手寫數字影像資料集，包含數字 0 至 9 的灰階影像，每張影像大小為 28 × 28 像素。

本案例使用 TensorFlow/Keras 載入 MNIST 資料集，建立神經網路模型，讓模型學習辨識不同的手寫數字。

資料集包含：

- 訓練資料：60,000 張影像。
- 測試資料：10,000 張影像。
- 影像尺寸：28 × 28 像素。
- 分類類別：0 至 9，共 10 類。

### 5.2 載入必要套件

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```

上述套件分別用於建立模型、進行數值運算，以及顯示影像。

### 5.3 載入 MNIST 資料集

```python
# 載入 MNIST 資料集
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# 將像素值正規化至 0～1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("訓練資料形狀：", x_train.shape)
print("測試資料形狀：", x_test.shape)
```

資料正規化可以將原本介於 0 至 255 的像素值縮放至 0 至 1，有助於模型進行訓練。

預期輸出：

```text
訓練資料形狀： (60000, 28, 28)
測試資料形狀： (10000, 28, 28)
```

### 5.4 顯示手寫數字影像

```python
plt.imshow(x_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.axis("off")
plt.show()
```

此程式會顯示第一張訓練影像，並在標題中顯示該影像對應的正確數字標籤。

### 5.5 建立神經網路模型

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
```

模型主要包含以下層：

- **Flatten：** 將 28 × 28 的影像轉換成一維資料。
- **Dense（128 個神經元）：** 學習影像中的特徵。
- **Dropout：** 降低模型過度依賴特定神經元的風險。
- **Dense（10 個神經元）：** 輸出數字 0 至 9 的分類結果。

### 5.6 訓練模型

```python
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)
```

模型使用訓練資料進行 5 個 Epoch 的訓練，並保留部分訓練資料作為驗證資料，以觀察模型的學習情況。

### 5.7 評估模型

```python
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=2
)

print(f"測試損失值：{test_loss:.4f}")
print(f"測試準確率：{test_accuracy:.4f}")
print(f"測試準確率百分比：{test_accuracy * 100:.2f}%")
```

透過測試資料評估模型，可以了解模型對未參與訓練的影像具有多少辨識能力。

實際準確率會依模型設定、訓練過程及執行環境而有所不同。

### 5.8 進行手寫數字預測

```python
# 選取一張測試影像
index = 0
image = x_test[index]

# 進行預測
prediction = model.predict(
    np.expand_dims(image, axis=0),
    verbose=0
)

predicted_label = np.argmax(prediction[0])
actual_label = y_test[index]

# 顯示影像與預測結果
plt.imshow(image, cmap="gray")
plt.title(
    f"Predicted: {predicted_label}, "
    f"Actual: {actual_label}"
)
plt.axis("off")
plt.show()

print("模型預測結果：", predicted_label)
print("實際數字標籤：", actual_label)
```

模型會輸出預測的數字，並與資料集提供的正確標籤進行比較，以觀察辨識結果是否正確。

## 六、執行結果與分析

完成模型訓練後，可以透過測試準確率及單張影像預測結果，觀察神經網路對手寫數字的辨識能力。

分析重點包括：

1. 模型是否能正確辨識大部分手寫數字。
2. 測試準確率是否符合預期。
3. 哪些手寫數字可能因筆畫相似而發生誤判。
4. 調整訓練次數、模型結構或其他參數後，辨識結果是否有所改變。

實際執行結果應以 Google Colab 中產生的數據與影像為準。

## 七、專案資料夾結構

```text
google-colab-mnist/
│
├── README.md
├── mnist_handwriting_recognition.ipynb
└── images/
    ├── colab_interface.png
    ├── mnist_sample.png
    └── training_result.png
```

檔案說明：

- `README.md`：專案介紹、操作教學及案例說明。
- `mnist_handwriting_recognition.ipynb`：完整的 Google Colab 實作程式。
- `images/`：存放操作介面、手寫數字與訓練結果的截圖。

以上為建議的專案結構，圖片及 Notebook 檔案需在實際建立後放入對應資料夾。

## 八、如何執行本專案

1. 開啟 Google Colab。
2. 將 `mnist_handwriting_recognition.ipynb` 上傳至 Colab，或將 Notebook 放入 Google Drive 後開啟。
3. 依序執行各個程式碼儲存格。
4. 等待模型完成訓練。
5. 查看測試準確率與手寫數字預測結果。

   <img width="1016" height="518" alt="image" src="https://github.com/user-attachments/assets/3d1abdba-2902-4aae-b464-f9c5c1fb938b" />

本案例使用的 TensorFlow、NumPy 與 Matplotlib 可在一般 Colab 執行環境中使用；若執行環境缺少相關套件，請依實際錯誤訊息安裝。

## 九、學習心得與結論

透過本專案，可以了解 Google Colab 的基本操作，以及如何使用 Python 與 TensorFlow 建立簡單的手寫數字辨識模型。

從資料載入、影像預處理、模型建立、訓練到評估，完整呈現機器學習的基本流程。藉由 MNIST 案例，也能初步認識神經網路在影像分類上的應用。

Google Colab 提供方便的雲端開發環境，適合初學者進行 Python 程式設計、人工智慧與機器學習的入門練習。


## 十、參考資料

- [Google Colab 官方網站](https://colab.research.google.com/)
- [TensorFlow 官方教學文件](https://www.tensorflow.org/tutorials)
- [Keras 官方文件](https://keras.io/)
- [MNIST 資料集說明](https://www.tensorflow.org/datasets/catalog/mnist)

---

**專案主題：** Google Colab 使用教學：附帶 MNIST 案例

**主要技術：** Python、Google Colab、TensorFlow、Keras

**專案類型：** 雲端程式設計與人工智慧入門實作
