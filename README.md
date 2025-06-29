## 环境搭建

```bash
uv sync
```

## 数据转换

```bash
uv run python NewData.py --dataset_name ceder
uv run python NewData.py --dataset_name retro
```

## 模型训练与测试

```bash
uv run --directory ElemwiseRetro python Train_P_custom.py --dataset_name ceder
uv run --directory ElemwiseRetro python Train_P_custom.py --dataset_name retro
```

## 实验结果

### ceder

top1: 2034/2934=0.693

top2: 2252/2934=0.768 

top3: 2309/2934=0.787 

top4: 2333/2934=0.795 

top5: 2343/2934=0.799 

### retro

top1: 1829/2558=0.715 

top2: 2011/2558=0.786 

top3: 2063/2558=0.806 

top4: 2086/2558=0.815 

top5: 2104/2558=0.823 
