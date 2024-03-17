import string
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2 as cv

from deep_text_recognition_benchmark.utils import CTCLabelConverter, AttnLabelConverter
from deep_text_recognition_benchmark.dataset import RawDataset, AlignCollate
from deep_text_recognition_benchmark.model import Model


class DTRB:
    def __init__(self, weights_path, opt):
        self.opt = opt
        if self.opt.sensitive:
            self.opt.character = string.printable[:-6]

        cudnn.benchmark = True
        cudnn.deterministic = True
        self.opt.num_gpu = torch.cuda.device_count()

        """ model configuration """
        if 'CTC' in self.opt.Prediction:
            self.converter = CTCLabelConverter(self.opt.character)
        else:
            self.converter = AttnLabelConverter(self.opt.character)
        self.opt.num_class = len(self.converter.character)

        if self.opt.rgb:
            self.opt.input_channel = 3

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(weights_path)

    def load_model(self, weights_path):
        self.model = Model(self.opt)
        print('model input parameters', self.opt.imgH, self.opt.imgW, self.opt.num_fiducial, self.opt.input_channel, self.opt.output_channel,
              self.opt.hidden_size, self.opt.num_class, self.opt.batch_max_length, self.opt.Transformation, self.opt.FeatureExtraction,
              self.opt.SequenceModeling, self.opt.Prediction)
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        # load model
        print('loading pretrained model from %s' % weights_path)
        self.model.load_state_dict(torch.load(
            weights_path, map_location=self.device))

    def predict(self, image) -> list[str, float]:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # predict
        self.model.eval()
        with torch.no_grad():
            image_tensor: torch.Tensor = transform(image)
            image_tensor = image_tensor.sub_(0.5).div_(0.5)
            image_tensor = torch.unsqueeze(
                image_tensor, 0)

            batch_size = image_tensor.size(0)
            image = image_tensor.to(self.device)

            length_for_pred = torch.IntTensor(
                [self.opt.batch_max_length] * batch_size).to(self.device)
            text_for_pred = torch.LongTensor(
                batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)

            if 'CTC' in self.opt.Prediction:
                preds = self.model(image, text_for_pred)

                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds_index = preds.max(2)[1]

                preds_str = self.converter.decode(preds_index, preds_size)

            else:
                preds = self.model(image, text_for_pred, is_train=False)

                preds_index = preds.max(2)[1]
                preds_str = self.converter.decode(preds_index, length_for_pred)

            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

            print(f'{dashed_line}\n{head}\n{dashed_line}')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(["besco"], preds_str, preds_max_prob):
                if 'Attn' in self.opt.Prediction:
                    pred_EOS = pred.find('[s]')

                    pred = pred[:pred_EOS]
                    pred_max_prob = pred_max_prob[:pred_EOS]

                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
        
        return pred, float(confidence_score)


if __name__ == "__main__":
    plate_recognizer = DTRB(
        "../weights/dtrb-recognizer/dtrb-None-VGG-BiLSTM-CTC-license-plate-recognizer.pth")

    image = cv.imread("../io/test.jpg")
    plate_recognizer.predict(image)
