# Text Emoji Classification
## This task solved with this repo
- Text Classification: I used [GloVe](https://nlp.stanford.edu/projects/glove/) pre-trained embeddings to convert my text to numerical vectors.

## How to install
### Run this command:
```bash
pip install -r requirements.txt
```
***Also you can download the GloVe pre-trained embeddings***
```bash
wget http://nlp.stanford.edu/data/glove.6B.zips
```

## How to train
### Run this command:
```bash
python train.py --train-dataset YOUR_TRAIN_DATASET --test-dataset YOUR_TEST_DATASET \
--dimension DIMENSION_OF_FEATURE_VECTORS --vectors-file YOUR_FEATURE_VECTORS_FILE \
--epochs NUMBER_OF_EPOCHS
```

#### You can also see the other arguments of it with this command
```bash
python train.py --help
```
*For Example:*
- *`--dropout, --no-dropout`*: You can add dropout layer to your network. **default:***`False`*
- *`--model-save`*: You change the best model name to save. **default:***`best_emojis_classifier.keras`*
- *`--save-plots, --no-save-plots`*: You can save the training information plots. **default:***`True`*

## How to test
### Run this command:
```bash
python test.py --model YOUR_MODEL --vectors-file YOUR_FEATURE_VECTORS_FILE \
--sentence YOUR_SENTENCE
```
#### You can also see the other arguments of it with this command
```bash
python test.py --help
```
*For Example:*
- *`--infer, --no-infer`*: Whether to inferences the model with your sentence or not. **default:***`True`*
- *`--n-infer`*: You can change number of inferences on your sentence. **default:***`100`*

## Benchmark
### Without Dropout layer
<table>
    <tr>
        <td>Featue Vector Dimension</td>
        <td>Train Loss</td>
        <td>Train Accuracy</td>
        <td>Test Loss</td>
        <td>Test Accuracy</td>
        <td>Inference Time</td>
    </tr>
    <tr>
        <td>50d</td>
        <td>0.3673</td>
        <td>0.9394</td>
        <td>0.4503</td>
        <td>0.8571</td>
        <td>0.0686s</td>
    </tr>    
    </tr>
        <td>100d</td>
        <td>0.3991</td>
        <td>0.9470</td>
        <td>0.4769</td>
        <td>0.8593</td>
        <td>0.0993s</td>
    </tr>
    <tr>
        <td>200d</td>
        <td>0.2039</td>
        <td>0.9848</td>
        <td>0.4449</td>
        <td>0.8214</td>
        <td>0.0721s</td>
    </tr>
    <tr>
        <td>300d</td>
        <td>0.1319</td>
        <td>0.9924</td>
        <td>0.4310</td>
        <td>0.8683</td>
        <td>0.0653s</td>
    </tr>
   
</table>

<br>

### With Dropout layer
<table>
    <tr>
        <td>Featue Vector Dimension</td>
        <td>Train Loss</td>
        <td>Train Accuracy</td>
        <td>Test Loss</td>
        <td>Test Accuracy</td>
        <td>Inference Time</td>
    </tr>
    <tr>
        <td>50d</td>
        <td>0.8322</td>
        <td>0.7273</td>
        <td>0.8891</td>
        <td>0.7321</td>
        <td>0.0671s</td>
    </tr>    
    </tr>
        <td>100d</td>
        <td>0.6902</td>
        <td>0.7955</td>
        <td>0.7373</td>
        <td>0.7679</td>
        <td>0.0773s</td>
    </tr>
    <tr>
        <td>200d</td>
        <td>0.5082</td>
        <td>0.9167</td>
        <td>0.5904</td>
        <td>0.8393</td>
        <td>0.0997s</td>
    </tr>
    <tr>
        <td>300d</td>
        <td>0.2764</td>
        <td>0.9697</td>
        <td>0.4969</td>
        <td>0.8750</td>
        <td>0.0639s</td>
    </tr>
   
</table>