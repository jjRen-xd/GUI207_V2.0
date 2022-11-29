
### 训练
```angular2html
python main.py --all_class=5 --batch_size=32 --bound=0.3 --increment_epoch=10 --learning_rate=0.001 --memory_size=200 --old_class=4 --pretrain_epoch=0 --random_seed=2022 --snr=2 --task_size=1 --test_ratio=0.5
```

### 测试
```angular2html
python evaluate.py --all_class=5 --batch_size=32 --bound=0.3 --increment_epoch=10 --learning_rate=0.001 --memory_size=200 --old_class=4 --pretrain_epoch=0 --random_seed=2022 --snr=2 --task_size=1 --test_ratio=0.5
```