# Source Code Summarization with StructuralRelative Position Guided Transformer
Official implementation of our SANER 2022 oral paper "Source Code Summarization with StructuralRelative Position Guided Transformer".

You can download the full dataset from the link https://drive.google.com/file/d/1mAxDZZ3eTffezOl7Fkzi8ey9ac3ePkEZ/view?usp=sharing .

## Quick Start

### Training
```
cd main
python train.py --dataset_name python --model_name YOUR_MODEL_NAME
``` 

### Logs are stored in 
```
../modelx/YOUR_MODEL_NAME.txt
```

### Testing
```
python test.py --dataset_name python --beam_size 5 --model_name YOUR_MODEL_NAME
```

