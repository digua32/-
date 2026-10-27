# Google Colab 使用教程：MNIST 手寫數字辨識

## 一、專案簡介

本專案旨在介紹 Google Colaboratory（Google Colab）的基本操作，並透過 MNIST 手寫數字辨識案例，示範如何使用 Python 與 TensorFlow 建立、訓練及評估人工智慧模型。

Google Colab 是一種雲端 Notebook 執行環境，使用者可以透過瀏覽器撰寫及執行 Python 程式，不必在自己的電腦上安裝完整的開發環境。

## 二、專案目的

1. 認識 Google Colab 的基本介面與操作方式。
2. 學習使用 TensorFlow 載入及處理 MNIST 資料集。
3. 建立簡單的類神經網路，辨識 0～9 的手寫數字。
4. 透過測試資料評估模型的辨識能力。

## 三、使用工具

- Google Colaboratory
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- GitHub

## 四、MNIST 資料集介紹

MNIST 是常見的手寫數字影像資料集，每張圖片為 28 × 28 像素的灰階影像，數字標籤介於 0～9。

本專案使用訓練資料讓模型學習手寫數字的特徵，再利用測試資料評估模型對未參與訓練之圖片的辨識能力。

## 五、實作流程

1. 開啟 Google Colab 並建立 Notebook。
2. 匯入 TensorFlow 等必要套件。
3. 載入 MNIST 訓練資料與測試資料。
4. 將圖片像素值正規化至 0～1。
5. 建立並編譯類神經網路模型。
6. 使用訓練資料進行模型訓練。
7. 使用測試資料評估模型準確率。
8. 顯示手寫數字圖片與模型預測結果。
9. 將 Notebook 儲存並上傳至 GitHub。

## 六、實驗結果

- 訓練輪數：5 epochs
- 測試準確率：（請填入實際執行結果）
- 預測結果：（請依實際執行結果說明）
- 實驗觀察：（請說明模型預測正確或錯誤的情況）

## 七、結論

本專案透過 Google Colab 執行 MNIST 手寫數字辨識案例，完成資料載入、資料預處理、模型建立、模型訓練與準確率評估等流程。

藉由本次實作，可以初步了解人工智慧影像分類的基本原理，以及如何運用雲端運算環境完成 AI 程式的執行與成果保存。

## 八、參考資料

1. TensorFlow 官方教學：
   https://www.tensorflow.org/tutorials/quickstart/beginner

2. Google Colab：
   https://colab.research.google.com/

3. GitHub：
   https://github.com/

## 九、組員資訊

- 組員一：（姓名）
- 組員二：（姓名）
