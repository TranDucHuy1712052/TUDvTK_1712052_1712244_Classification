# TUDvTK_1712052_1712242_Classification
(EDIT: MSSV2 = 1712242, do người tạo repo có sự nhầm lẫn)
Đồ án cuối kì môn Toán ứng dụng và thống kê của nhóm 1712052_1712242

## Pretrain model:
https://drive.google.com/drive/folders/1VLGHYp-5n9FqAW1IlRfSapl4FsOVIa5T

## Hướng dẫn sử dụng: 
- Tải dữ liệu xuống (ở đây nhóm xài dữ liệu Adult Data Set - UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Adult)

- Sửa lại dữ liệu như sau 
(hoặc là tải xuống qua đây để không mất thời gian: https://drive.google.com/drive/folders/1LK3vWO0ZdfGUPI2On-9AaJQ6t1i6qaQ5?usp=sharing)
  - Sao chép file dữ liệu gốc (adult.data) sang một tập tin khác, nhưng dưới dạng tập tin csv. Đặt tên là adult.train.csv. Tương tự với bộ dữ liệu test (đặt tên adult.test.csv)
  - Trong các tập tin csv, thêm một hàng là tên các cột (thuộc tính). Cụ thể là
'age	workclass	fnlwgt	education	education-num	marital-status	occupation	relationship	race	sex	capital-gain	capital-loss	hours-per-week	native-country	label'
  - Với file adult.names: Bỏ hết nội dung file trừ các dòng mô tả thuộc tính (từ 'age: continuous.' trở xuống).
  
- Sửa lại đường dẫn trong code thành đường dẫn tới dữ liệu. Lưu ý code của nhóm dùng thư viện Pandas, đọc bằng tập tin .csv

- Gõ lệnh dưới đây để chạy thử:
```
python main.py
```

- Nếu muốn thay đổi mô hình SVM, xin hãy chỉnh sửa lại mô hình khác tại dòng thứ 15 của tập tin “models/svm.py”
```
class SVMClassifier :
    def __init__ ( self ):
    ...
    self .model = svm.SVC()     ## thay đổi mô hình tại đây
```


