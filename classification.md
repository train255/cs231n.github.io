---
layout: page
mathjax: true
permalink: /classification/
---

Bài giảng này nhằm giới thiệu về vấn đề phân loại hình ảnh (Image Classification) và phương pháp tiếp cận dữ liệu (data-driven approach). Mục lục:

- [Intro to Image Classification, data-driven approach, pipeline](#intro)
- [Nearest Neighbor Classifier](#nn)
  - [k-Nearest Neighbor](#knn)
- [Validation sets, Cross-validation, hyperparameter tuning](#val)
- [Pros/Cons of Nearest Neighbor](#procon)
- [Summary](#summary)
- [Summary: Applying kNN in practice](#summaryapply)
- [Further Reading](#reading)

<a name='intro'></a>

## Image Classification

**Motivation**. Trong phần này chúng ta sẽ giới thiệu về bài toán phân loại ảnh, nhiệm vụ của chúng ta là sẽ sẽ gán nhãn cho một bức ảnh đầu vào từ một tập các nhãn cố định. Đây là một vấn đề cốt lõi của Computer Vision, mặc dù đơn giản nhưng sẽ ứng dụng nhiều trong thực tế. Hơn nữa, sau khóa học chúng ta sẽ thấy nhiều bài toán khác Computer Vision, chẳng hạn như nhận dạng đối tượng (object detection), các bài toán gom nhóm (segmentation),...cũng có thể đưa về bài toán phân loại hình ảnh.

**Example**. Ví dụ, như hình dưới đây là một mô hình phân loại ảnh để chỉ ra một ảnh là nhãn nào trong 4 nhãn *{cat, dog, hat, mug}*. Lưu ý rằng đối với máy tính chỉ hiểu được tín hiệu số nên như trong ảnh một bức ảnh sẽ được biểu diễn dưới dạng một mảng 3 chiều các chữ số. Như trong ví dụ, là hình ảnh một con mèo có kích thước là 248 x 400 (pixel), và có ba màu là Red,Green,Blue (RGB). Do đó hình ảnh sẽ bao gồm 248 x 400 x 3 = 297,600 chữ số. Mỗi chữ số là một số nguyên nằm trong khoảng từ 0 (đen) đến 255 (trắng). Nhiệm vụ của chúng ta là chuyển từ hàng triệu những con số đó để gán thành một nhãn, chẳng hạn *"cat"*.

<div class="fig figcenter fighighlight">
  <img src="/assets/classify.png">
  <div class="figcaption">Nhiệm vụ của Image Classification là dự đoán một nhãn (hoặc như ở đây chúng ta hiển thị tỷ lệ phần trăm các nhãn) cho một bức ảnh. Ảnh là một mảng 3 chiều các số nguyên từ 0 đến 255 có kích thước là Width x Height x 3. Số 3 ở đây đại diện cho ba màu Red, Green, Blue.</div>
</div>

**Challenges**. Việc nhận dạng hình ảnh (chẳng hạn như ảnh mèo) đối với con người là một điều tầm thường, nhưng nó là thách thức trên quan điểm của một thuật toán Computer Vision. Khi chúng tôi trình bày danh sách các thách thức dưới đây, hãy nhớ rằng sự đại diện thô của hình ảnh giống như một mảng 3 chiều của các giá trị độ sáng:

- **Viewpoint variation**. Một thể hiện của một đối tượng có thể được chụp từ nhiều góc nhìn khác nhau của máy ảnh.
- **Scale variation**. Các nhóm (class) hình ảnh thường có sự khác biệt về kích cỡ.
- **Deformation**. Nhiều đối tượng không phải là một cơ thể cứng nhắc mà nó có thể bị biến dạng.
- **Occlusion**. Các đối tượng có thể bị che khuất. Đôi khi chỉ nhìn thấy một phần nhỏ của đối tượng (một vài pixel).
- **Illumination conditions**. Ảnh hưởng của độ sáng tác động lên các giá trị pixel.
- **Background clutter**. Các đối tượng có thể bị trộn lẫn vào môi trường xung quanh làm cho chúng khó xác định.
- **Intra-class variation**. Các nhóm hình ảnh có thể tương đối rộng, chẳng hạn như nhận dạng *ghế*. Có rất nhiều loại ghế và mỗi loại ghế lại có hình dạng khác nhau.

Một model phân loại hình ảnh tốt phải không bị thay đổi với các biến thể nói trên, nhưng vẫn phải giữ được độ nhạy với các ảnh bình thường không thuộc các biến thể nói trên [inter-class variation](http://journals.plos.org/plosone/article/figure/image?size=medium&id=10.1371/journal.pone.0099212.g001).

<div class="fig figcenter fighighlight">
  <img src="/assets/challenges.jpeg">
  <div class="figcaption"></div>
</div>

**Data-driven approach**. Làm sao có thể viết một thuật toán phân loại hình ảnh thành các nhóm khác nhau? Không giống như việc chúng ta viết những thuật toán khác, chẳng hạn như, thuật toán sắp xếp danh sách các chữ số, nó không rõ ràng để chúng ta có thể viết một thuật toán xác định mèo trong hình ảnh. Do đó thay vì cố gắng phân loại hình ảnh trực tiếp trong code chúng ta sẽ tiếp cận theo cách sau: chúng ta sẽ cung cấp cho máy tính nhiều ví dụ cho mỗi nhóm và chúng ta sẽ phát triển những thuật toán học trên những ví dụ này để tìm hiểu sự xuất hiện của các hình ảnh trong mỗi nhóm. Cách tiếp cận này gọi là *phương pháp tiếp cận dữ liệu* (data-driven approach), vì nó dựa vào sự tích lũy đầu tiên trên *tập dữ liệu huấn luyện (training dataset)* của các ảnh đã được gán nhãn. Dưới đây là ví dụ về bộ dữ liệu huấn luyện:

<div class="fig figcenter fighighlight">
  <img src="/assets/trainset.jpg">
  <div class="figcaption">Một ví dụ về tập huấn luyện cho bốn loại hình ảnh. Trong thực tế chúng ta có thể có hàng nghìn loại hình ảnh và hàng trăm nghìn hình ảnh cho mỗi loại.</div>
</div>

**The image classification pipeline**. Chúng ta đã biết nhiệm vụ của phân loại hình ảnh là chúng ta sẽ tạo ra một ma trận của các pixel để đại diện cho một bức ảnh và gán nhãn cho nó một nhãn. Quá trình của công việc này sẽ được mô tả như dưới đây:

- **Input:** Đầu vào của chúng ta bao gồm một tập *N* các hình ảnh, và mỗi hình ảnh được gán một nhãn trong tập *K* các nhãn khác nhau. Chúng ta gọi tập dữ liệu này là *tập huấn luyện (training set)*.
- **Learning:** Nhiệm vụ của chúng ta là sử dụng tập huấn luyện để học xem mỗi hình ảnh thuộc một nhãn sẽ trông như thế nào. Chúng ta gọi bước này là *đào tạo một trình phân loại (training a classifier)*, hay *học một mô hình (learning a model)*.
- **Evaluation:** Cuối cùng, chúng tôi đánh giá chất lượng classifier bằng cách cho nó đoán nhãn của một hình ảnh mới mà nó chưa từng thấy trước đây. Chúng ta sẽ so sánh nhãn thực sự của bức ảnh và nhãn mà classifier dự đoán. Theo trực giác, chúng ta hy vọng sẽ có nhiều nhãn dự đoán trùng khớp với những nhãn thực sự (ground truth) (khái niệm *ground truth* hiểu đơn giản chính là nhãn/label/đầu ra thực sự của các điểm trong dữ liệu test).

<a name='nn'></a>

### Nearest Neighbor Classifier
Cách tiếp cận đầu tiên chúng ta sẽ tìm hiểu về **Nearest Neighbor Classifier**. Classifier này không liên quan đến Convolutional Neural Networks và rất hiếm khi được sử dụng trong thực tế, nhưng nó cho phép chúng ta có cái nhìn cơ bản về cách giải quyết bài toán phân loại hình ảnh. 

**Example image classification dataset: CIFAR-10.** Một bộ dữ liệu phân loại ảnh phổ biến đó là <a href="http://www.cs.toronto.edu/~kriz/cifar.html">bộ dữ liệu CIFAR-10</a>. Bộ dữ liệu này bao gồm 60,000 hình ảnh nhỏ có kích thước 32x32 pixel. Mỗi ảnh được gán một nhãn trong số 10 nhãn (ví dụ *"airplane, automobile, bird, etc"*). 60,000 ảnh này được chia thành 2 loại bao gồm 50,000 hình ảnh cho tập huấn luyện 10,000 hình ảnh cho tập thử nghiệm. Như trong ảnh dưới đây bạn có thể thấy 10 ảnh ví dụ ngẫu nhiên từ mỗi một nhóm trong 10 nhóm:

<div class="fig figcenter fighighlight">
  <img src="/assets/nn.jpg">
  <div class="figcaption">Trái: Các ảnh ví dụ từ <a href="http://www.cs.toronto.edu/~kriz/cifar.html">bộ dữ liệu CIFAR-10</a>. Phải: cột đầu tiên hiển thị một vài ảnh thử nghiệm và các cột tiếp theo là 10 ảnh hàng xóm gần nhất (nearest neighbors) trong tập huấn luyện dựa theo sự khác biệt về pixel.</div>
</div>

Giả sử bây giờ chúng ta có tập huấn luyện CIFAR-10 bao gồm 50,000 hình ảnh (5,000 ảnh cho mỗi nhãn), và chúng ta muốn gán nhãn cho 10,000 bức ảnh còn lại. Nearest neighbor classifier sẽ lấy một ảnh trong tập thử nghiệm, so sánh nó với từng ảnh trong tập huấn luyện, và dựa vào ảnh huấn luyện sát nhất để dự đoán nhãn. Như ta thấy ở trên có thể thấy kết quả của 10 bức ảnh thử nghiệm. Chú ý rằng chỉ có 3 trong số 10 ví dụ, một hình ảnh của cùng một nhóm được tìm thấy, trong khi 7 ví dụ khác thì không như vậy. Ví dụ, ở hàng thứ 8 ảnh huấn luyện gần nhất với ảnh đầu con ngựa là một chiếc xe màu đỏ, có thể là do nền màu đen đã ảnh hưởng đến kết quả. Kết quả là, hình ảnh con ngựa trong trường hợp này đã bị nhầm lẫn sang một chiếc xe hơi.

Bạn có thể thấy rằng chúng tôi đã không xác định một cách chi tiết cách mà chúng tôi so sánh hai bức ảnh, trong trường hợp này chỉ là 2 khối có kích thưóc 32 x 32 x 3. Một trong những khả năng đơn giản nhất là so sánh từng pixel của 2 bức ảnh và tìm tất cả sự khác nhau giữa hai bức ảnh. Nói cách khác, ta sẽ biểu diễn ảnh dưới dạng vector \\( I_1, I_2 \\) , một sự lựa chọn hợp lý để so sánh chúng là tính khoảng cách 2 vector đó **L1 distance**:

$$
d_1 (I_1, I_2) = \sum_{p} \left| I^p_1 - I^p_2 \right|
$$

Tổng được tính qua tất cả các điểm ảnh, chúng ta có thể hình dung việc tính khoảng cách như hình dưới đây:

<div class="fig figcenter fighighlight">
  <img src="/assets/nneg.jpeg">
  <div class="figcaption">Một ví dụ sử dụng phương pháp [pixel-wise](https://support.pcigeomatics.com/hc/en-us/article_attachments/202615176/image001.png) (so sánh từng cặp pixel ở vị trí tương ứng trên 2 ảnh) để so sánh 2 ảnh với khoảng cách L1 (trong ví dụ này ta chỉ xét trên một kênh màu trong 3 màu R, G, B). Chúng ta sẽ thực hiện từng phép trừ trên 2 pixel tương ứng trên hai ảnh và kết quả sẽ được lưu lại vào một block có cùng kích thước. Nếu hai ảnh giống nhau thì kết quả sẽ là 0. Còn nếu 2 ảnh rất khác nhau thì kết quả sẽ lớn.</div>
</div>

Chúng ta cũng sẽ tìm hiểu cách lập trình ra classifier. Đầu tiên, chúng ta tải dữ liệu CIFAR-10 và lưu vào trong 4 mảng: các nhãn huấn luyện, các dữ liệu huấn luyện, các nhãn kiểm thử, các dữ liệu kiểm thử. Như ở code dưới đây, `Xtr` (có kích thước 50,000 x 32 x 32 x 3) chứa toàn bộ ảnh trong tập huấn luyện, và tương ứng ta có mảng 1 chiều `Ytr` (độ dài 50,000) chứa toàn bộ nhãn huấng luyện (từ 0 tới 9):

```python
Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
```

Bây giờ chúng ta đã có tất cả hình theo hàng, và dưới đây sẽ là cách chúng ta huấn luyện và dự đoán một phân loại:

```python
nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )
```

Lưu ý rằng theo tiêu chí đánh giá, phổ biến nhất là sử dụng **độ chính xác (accuracy)**, để đo độ chính xác của việc dự đoán. Lưu ý rằng tất cả các classifier chúng ta xây dựng sẽ đáp ứng một API chung: chúng bao gồm một hàm `train(X,y)` lấy dữ liệu và các nhãn để học. Bên trong classifier nên xây dựng một số loại mô hình nhãn và cách thức để có thể dự đoán từ dữ liệu. Và sau đó sẽ có một hàm `predict(X)`, nó sẽ lấy dữ liệu mới và dự đoán các nhãn:

```python
import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in xrange(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred
```

Nếu bạn chạy đoạn code này, bạn sẽ thấy trình phân loại này chỉ đạt **38.6%** trên CIFAR-10. Tỷ lệ tốt hơn so với việc đoán ngẫu nhiên (với 10 nhóm nếu đoán ngẫu nhiên thì tỷ lệ đoán đúng là 10%), nhưng vẫn còn kém hơn so với tỷ lệ mà con người dự đoán ([ước tính khoảng 94%](http://karpathy.github.io/2011/04/27/manually-classifying-cifar10/)) hoặc so với kỹ thuật tiên tiến Convolutional Neural Networks đạt được 95%, gần với độ chính xác như con người (xem [bảng xếp hạng](http://www.kaggle.com/c/cifar-10/leaderboard) của cuộc thi Kaggle trên CIFAR-10).

**The choice of distance.** 
Có rất nhiều cách để tính toán khoảng cách giữa các vector. Một sự lựa chọn phổ biến khác nữa là **L2 distance**, nó chính là khoảng cách Euclid giữa 2 vector. Khoảng cách Euclid được tính như sau:

$$
d_2 (I_1, I_2) = \sqrt{\sum_{p} \left( I^p_1 - I^p_2 \right)^2}
$$

Nói cách khác chúng ta sẽ tính toán sự khác biệt pixel (pixelwise) như trước, nhưng lần này chúng ta sẽ tính bình phương của chúng, sau đó lấy căn bậc hai. Như trong đoạn code dưới đây ta sử dụng thư viện numpy:

```python
distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
```

Lưu ý rằng tôi đã thêm `np.sqrt` ở trong đoạn code phía trên, nhưng trong thực tế với nearest neighbor chúng ta có thể bỏ phần căn bậc hai bởi vì nó là một *hàm đơn điệu (monotonic function)*. Nếu việc tính khoảng cách chỉ để phục vụ việc sắp xếp thì ta không cần đến bước căn bậc hai. Nếu bạn chạy đoạn code trên với CIFAR-10, bạn sẽ đạt được độ chính xác **35.4%** (thấp hơn một chút so với kết quả L1).

**L1 vs. L2.** Có một điều thú vị khi xem xét sự khác nhau giữa hai metric này. Đặc biệt khoảng cách L2 cho tỷ lệ thấp hơn so với L1 khi so sánh khoảng cách hai vector. Khoảng cách L1 và L2 (hay các tiêu chuẩn (norm) L1, L2 về sự khác nhau giữa một cặp hình ảnh) là các trường hợp đặc biệt được sử dụng phổ biến nhất của [p-norm](http://planetmath.org/vectorpnorm).

<a name='knn'></a>

### k - Nearest Neighbor Classifier

Bạn có thể nhận thấy rằng thật lạ là chỉ sử dụng một nhãn gần với ảnh nhất khi chúng ta dự đoán. Thật vậy, gần như luôn có một cách để làm tốt hơn và trong trường hợp này ta có **k-Nearest Neighbor Classifier**. Ý tưởng rất đơn giản: thay vì tìm kiếm một ảnh gần nhất trong tập huấn luyện, chúng ta sẽ tìm top **k** các ảnh gần nhất, và chúng ta sẽ bỏ phiếu chọn ra nhãn của tập ảnh thử nghiệm. Với *k = 1*, ta sẽ quay trở lại bài toán Nearest Neighbor classifier. Với **k** càng cao thì sẽ giúp cho classifier bỏ được nhiều ngoại lệ:

<div class="fig figcenter fighighlight">
  <img src="/assets/knn.jpeg">
  <div class="figcaption">Một ví dụ về sự khác nhau giữa Nearest Neighbor và 5-Nearest Neighbor classifier, sử dụng mảng hai chiều và 3 lớp màu (red, blue, green). Các <b>vùng nền (decision boundaries)</b> thể hiện các điểm được phân loại vào lớp có màu tương ứng khi sử dụng khoảng cách L2. Các vùng màu trắng thể hiện các điểm phân loại không rõ ràng (ví dụ có ít nhất 2 nhãn có cùng số phiếu bầu). Lưu ý rằng trong trường hợp NN classifier, các điểm dữ liệu ngoại lệ (ví dụ điểm màu xanh lá nằm giữa một đám mấy các điểm màu xanh lục) tạo ra các hòn đào nhỏ có khả năng dự đoán sai, trong khi đó với 5-NN classifier nó sẽ làm trơn những phần này, làm cho <b>sự tổng quát hóa</b> dữ liệu được tốt hơn trên tập dữ liệu thử nghiệm. Cũng lưu ý rằng các vùng màu xám trong 5-NN được tạo ra bởi các liên kết trong phiếu bầu giữa các hàng xóm gần nhất (ví dụ 2 neighbor màu đỏ, tiếp đến là 2 neighbor màu xanh lục, cuối cùng là neighbor màu xanh lá).</div>
</div>

In practice, you will almost always want to use k-Nearest Neighbor. But what value of *k* should you use? We turn to this problem next.

<a name='val'></a>

### Validation sets for Hyperparameter tuning

The k-nearest neighbor classifier requires a setting for *k*. But what number works best? Additionally, we saw that there are many different distance functions we could have used: L1 norm, L2 norm, there are many other choices we didn't even consider (e.g. dot products). These choices are called **hyperparameters** and they come up very often in the design of many Machine Learning algorithms that learn from data. It's often not obvious what values/settings one should choose.

You might be tempted to suggest that we should try out many different values and see what works best. That is a fine idea and that's indeed what we will do, but this must be done very carefully. In particular, **we cannot use the test set for the purpose of tweaking hyperparameters**. Whenever you're designing Machine Learning algorithms, you should think of the test set as a very precious resource that should ideally never be touched until one time at the very end. Otherwise, the very real danger is that you may tune your hyperparameters to work well on the test set, but if you were to deploy your model you could see a significantly reduced performance. In practice, we would say that you **overfit** to the test set. Another way of looking at it is that if you tune your hyperparameters on the test set, you are effectively using the test set as the training set, and therefore the performance you achieve on it will be too optimistic with respect to what you might actually observe when you deploy your model. But if you only use the test set once at end, it remains a good proxy for measuring the **generalization** of your classifier (we will see much more discussion surrounding generalization later in the class).

> Evaluate on the test set only a single time, at the very end.

Luckily, there is a correct way of tuning the hyperparameters and it does not touch the test set at all. The idea is to split our training set in two: a slightly smaller training set, and what we call a **validation set**. Using CIFAR-10 as an example, we could for example use 49,000 of the training images for training, and leave 1,000 aside for validation. This validation set is essentially used as a fake test set to tune the hyper-parameters.

Here is what this might look like in the case of CIFAR-10:

```python
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:
  
  # use a particular value of k and evaluation on validation data
  nn = NearestNeighbor()
  nn.train(Xtr_rows, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predict(Xval_rows, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))
```

By the end of this procedure, we could plot a graph that shows which values of *k* work best. We would then stick with this value and evaluate once on the actual test set.

> Split your training set into training set and a validation set. Use validation set to tune all hyperparameters. At the end run a single time on the test set and report performance.

**Cross-validation**.
In cases where the size of your training data (and therefore also the validation data) might be small, people sometimes use a more sophisticated technique for hyperparameter tuning called **cross-validation**. Working with our previous example, the idea is that instead of arbitrarily picking the first 1000 datapoints to be the validation set and rest training set, you can get a better and less noisy estimate of how well a certain value of *k* works by iterating over different validation sets and averaging the performance across these. For example, in 5-fold cross-validation, we would split the training data into 5 equal folds, use 4 of them for training, and 1 for validation. We would then iterate over which fold is the validation fold, evaluate the performance, and finally average the performance across the different folds.

<div class="fig figleft fighighlight">
  <img src="/assets/cvplot.png">
  <div class="figcaption">Example of a 5-fold cross-validation run for the parameter <b>k</b>. For each value of <b>k</b> we train on 4 folds and evaluate on the 5th. Hence, for each <b>k</b> we receive 5 accuracies on the validation fold (accuracy is the y-axis, each result is a point). The trend line is drawn through the average of the results for each <b>k</b> and the error bars indicate the standard deviation. Note that in this particular case, the cross-validation suggests that a value of about <b>k</b> = 7 works best on this particular dataset (corresponding to the peak in the plot). If we used more than 5 folds, we might expect to see a smoother (i.e. less noisy) curve.</div>
  <div style="clear:both"></div>
</div>


**In practice**. In practice, people prefer to avoid cross-validation in favor of having a single validation split, since cross-validation can be computationally expensive. The splits people tend to use is between 50%-90% of the training data for training and rest for validation. However, this depends on multiple factors: For example if the number of hyperparameters is large you may prefer to use bigger validation splits. If the number of examples in the validation set is small (perhaps only a few hundred or so), it is safer to use cross-validation. Typical number of folds you can see in practice would be 3-fold, 5-fold or 10-fold cross-validation.

<div class="fig figcenter fighighlight">
  <img src="/assets/crossval.jpeg">
  <div class="figcaption">Common data splits. A training and test set is given. The training set is split into folds (for example 5 folds here). The folds 1-4 become the training set. One fold (e.g. fold 5 here in yellow) is denoted as the Validation fold and is used to tune the hyperparameters. Cross-validation goes a step further and iterates over the choice of which fold is the validation fold, separately from 1-5. This would be referred to as 5-fold cross-validation. In the very end once the model is trained and all the best hyperparameters were determined, the model is evaluated a single time on the test data (red).</div>
</div>

<a name='procon'></a>

**Pros and Cons of Nearest Neighbor classifier.**

It is worth considering some advantages and drawbacks of the Nearest Neighbor classifier. Clearly, one advantage is that it is very simple to implement and understand. Additionally, the classifier takes no time to train, since all that is required is to store and possibly index the training data. However, we pay that computational cost at test time, since classifying a test example requires a comparison to every single training example. This is backwards, since in practice we often care about the test time efficiency much more than the efficiency at training time. In fact, the deep neural networks we will develop later in this class shift this tradeoff to the other extreme: They are very expensive to train, but once the training is finished it is very cheap to classify a new test example. This mode of operation is much more desirable in practice.

As an aside, the computational complexity of the Nearest Neighbor classifier is an active area of research, and several **Approximate Nearest Neighbor** (ANN) algorithms and libraries exist that can accelerate the nearest neighbor lookup in a dataset (e.g. [FLANN](http://www.cs.ubc.ca/research/flann/)). These algorithms allow one to trade off the correctness of the nearest neighbor retrieval with its space/time complexity during retrieval, and usually rely on a pre-processing/indexing stage that involves building a kdtree, or running the k-means algorithm.

The Nearest Neighbor Classifier may sometimes be a good choice in some settings (especially if the data is low-dimensional), but it is rarely appropriate for use in practical image classification settings. One problem is that images are high-dimensional objects (i.e. they often contain many pixels), and distances over high-dimensional spaces can be very counter-intuitive. The image below illustrates the point that the pixel-based L2 similarities we developed above are very different from perceptual similarities:

<div class="fig figcenter fighighlight">
  <img src="/assets/samenorm.png">
  <div class="figcaption">Pixel-based distances on high-dimensional data (and images especially) can be very unintuitive. An original image (left) and three other images next to it that are all equally far away from it based on L2 pixel distance. Clearly, the pixel-wise distance does not correspond at all to perceptual or semantic similarity.</div>
</div>

Here is one more visualization to convince you that using pixel differences to compare images is inadequate. We can use a visualization technique called <a href="http://homepage.tudelft.nl/19j49/t-SNE.html">t-SNE</a> to take the CIFAR-10 images and embed them in two dimensions so that their (local) pairwise distances are best preserved. In this visualization, images that are shown nearby are considered to be very near according to the L2 pixelwise distance we developed above:

<div class="fig figcenter fighighlight">
  <img src="/assets/pixels_embed_cifar10.jpg">
  <div class="figcaption">CIFAR-10 images embedded in two dimensions with t-SNE. Images that are nearby on this image are considered to be close based on the L2 pixel distance. Notice the strong effect of background rather than semantic class differences. Click <a href="/assets/pixels_embed_cifar10_big.jpg">here</a> for a bigger version of this visualization.</div>
</div>

In particular, note that images that are nearby each other are much more a function of the general color distribution of the images, or the type of background rather than their semantic identity. For example, a dog can be seen very near a frog since both happen to be on white background. Ideally we would like images of all of the 10 classes to form their own clusters, so that images of the same class are nearby to each other regardless of irrelevant characteristics and variations (such as the background). However, to get this property we will have to go beyond raw pixels.

<a name='summary'></a>

### Summary

In summary:

- We introduced the problem of **Image Classification**, in which we are given a set of images that are all labeled with a single category. We are then asked to predict these categories for a novel set of test images and measure the accuracy of the predictions.
- We introduced a simple classifier called the **Nearest Neighbor classifier**. We saw that there are multiple hyper-parameters (such as value of k, or the type of distance used to compare examples) that are associated with this classifier and that there was no obvious way of choosing them.
-  We saw that the correct way to set these hyperparameters is to split your training data into two: a training set and a fake test set, which we call **validation set**. We try different hyperparameter values and keep the values that lead to the best performance on the validation set.
- If the lack of training data is a concern, we discussed a procedure called **cross-validation**, which can help reduce noise in estimating which hyperparameters work best.
- Once the best hyperparameters are found, we fix them and perform a single **evaluation** on the actual test set.
- We saw that Nearest Neighbor can get us about 40% accuracy on CIFAR-10. It is simple to implement but requires us to store the entire training set and it is expensive to evaluate on a test image. 
- Finally, we saw that the use of L1 or L2 distances on raw pixel values is not adequate since the distances correlate more strongly with backgrounds and color distributions of images than with their semantic content.

In next lectures we will embark on addressing these challenges and eventually arrive at solutions that give 90% accuracies, allow us to completely discard the training set once learning is complete, and they will allow us to evaluate a test image in less than a millisecond.

<a name='summaryapply'></a>

### Summary: Applying kNN in practice

If you wish to apply kNN in practice (hopefully not on images, or perhaps as only a baseline) proceed as follows:

1. Preprocess your data: Normalize the features in your data (e.g. one pixel in images) to have zero mean and unit variance. We will cover this in more detail in later sections, and chose not to cover data normalization in this section because pixels in images are usually homogeneous and do not exhibit widely different distributions, alleviating the need for data normalization.
2. If your data is very high-dimensional, consider using a dimensionality reduction technique such as PCA ([wiki ref](http://en.wikipedia.org/wiki/Principal_component_analysis), [CS229ref](http://cs229.stanford.edu/notes/cs229-notes10.pdf), [blog ref](http://www.bigdataexaminer.com/understanding-dimensionality-reduction-principal-component-analysis-and-singular-value-decomposition/)) or even [Random Projections](http://scikit-learn.org/stable/modules/random_projection.html).
3. Split your training data randomly into train/val splits. As a rule of thumb, between 70-90% of your data usually goes to the train split. This setting depends on how many hyperparameters you have and how much of an influence you expect them to have. If there are many hyperparameters to estimate, you should err on the side of having larger validation set to estimate them effectively. If you are concerned about the size of your validation data, it is best to split the training data into folds and perform cross-validation. If you can afford the computational budget it is always safer to go with cross-validation (the more folds the better, but more expensive).
4. Train and evaluate the kNN classifier on the validation data (for all folds, if doing cross-validation) for many choices of **k** (e.g. the more the better) and across different distance types (L1 and L2 are good candidates)
5. If your kNN classifier is running too long, consider using an Approximate Nearest Neighbor library (e.g. [FLANN](http://www.cs.ubc.ca/research/flann/)) to accelerate the retrieval (at cost of some accuracy).
6. Take note of the hyperparameters that gave the best results. There is a question of whether you should use the full training set with the best hyperparameters, since the optimal hyperparameters might change if you were to fold the validation data into your training set (since the size of the data would be larger). In practice it is cleaner to not use the validation data in the final classifier and consider it to be *burned* on estimating the hyperparameters. Evaluate the best model on the test set. Report the test set accuracy and declare the result to be the performance of the kNN classifier on your data.

<a name='reading'></a>

#### Further Reading

Here are some (optional) links you may find interesting for further reading:

- [A Few Useful Things to Know about Machine Learning](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf), where especially section 6 is related but the whole paper is a warmly recommended reading.

- [Recognizing and Learning Object Categories](http://people.csail.mit.edu/torralba/shortCourseRLOC/index.html), a short course of object categorization at ICCV 2005.
