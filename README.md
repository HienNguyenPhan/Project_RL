# Project_RL

Project Reinforcement Learning cuối kì của nhóm 35

Thành viên nhóm: Nguyễn Phan Hiển

## Phương pháp sử dụng
Deep Q Learning

## Cách chạy code
Cách chạy hàm train
```
Jupyter Notebook Project_RL.ipynb
```
Cách chạy hàm eval
```
python eval.py
```

## Kết quả
Video khi test với agent random

![random](https://github.com/user-attachments/assets/480ae032-0b36-4a6b-ac98-a24efb6472f6)

Video khi test với agent pre-train

![pretrained](https://github.com/user-attachments/assets/4c0e13e2-50ea-4c58-98aa-cccf96e20e6e)

Video khi test với agent pre-train final

![final_pretrained](https://github.com/user-attachments/assets/b97db9fc-5bb5-4b86-b874-35a9e92c55c0)

Video được tạo bởi main.py
Kết quả chiến thắng trong cả 3 ván. Tuy nhiên có trường hợp hòa do không đủ thời gian để tấn công hoặc còn sót lại một vài agent địch cuối cùng ở góc battlefield.

### Kết quả Eval

| Agent           | Winrate       | Reward_red | Reward_blue |
| ----------------|:-------------:|:----------:|------------:|
| Random          | 1             |   -3.32    |    2.14     |
| Pre_train       | 1             |    0.83    |    3.9      |
| Pre_train_final | 1             |    1.34    |    1.38     |
