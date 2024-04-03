from time import time
from argparse import ArgumentParser, BooleanOptionalAction

from text_classifier import EmojiTextClassifier

args = ArgumentParser()
args.add_argument('--model', type=str,
                  required=True, help='The model path')
args.add_argument('--vectors-file', type=str,
                  default='./glove_6B/glove.6B.200d.txt', help='The feature vectors file path')
args.add_argument('--sentence', type=str,
                  required=True, help='The sentence to test the model')
args.add_argument('--infer', type=bool, default=True,
                  action=BooleanOptionalAction, help='Whether to inferences the model with your sentence or not')
args.add_argument('--num-infer', type=int,
                  default=100, help='Number of inferences on your sentence')

opt = args.parse_args()
classifier = EmojiTextClassifier(...)
classifier.load_feature_vectors(opt.vectors_file)
classifier.load_model(opt.model)
emoji = classifier.predict(opt.sentence)
print(emoji)

if opt.infer:
    start_time = time()
    for i in range(opt.num_infer):
        classifier.predict(opt.sentence)
    
    duration = time() - start_time
    print(f'\nAverage inference time: {duration / opt.num_infer}')
