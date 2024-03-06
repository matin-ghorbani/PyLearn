import os
from lxml import etree as et
from lxml.etree import _Element
from tqdm import tqdm

IMAGES_DATASET: str = 'my_images'
GT_PATH: str = 'gt_train.txt'

words_dict: dict[str, str] = {
    'A': 'الف',
    'B': 'ب',
    'P': 'پ',
    'T': 'ت',
    'Y': 'ث',
    'Z': 'ز',
    'X': 'ش',
    'E': 'ع',
    'F': 'ف',
    'K': 'ک',
    'G': 'گ',
    'D': 'D',
    'S': 'S',
    'J': 'ج',
    'W': 'د',
    'C': 'س',
    'U': 'ص',
    'R': 'ط',
    'Q': 'ق',
    'L': 'ل',
    'M': 'م',
    'N': 'ن',
    'V': 'و',
    'H': 'ه',
    'I': 'ی',
    '0': '0',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    '@': 'ویلچر',
}


def prepare_dataset(images_dataset_path: str, dataset_type: str) -> None:
    gt_file = open(f'gt_{dataset_type}.txt', 'w', encoding='utf-8')
    parser = et.XMLParser(encoding='utf-8')

    root: _Element

    for file in tqdm(os.listdir(images_dataset_path)):
        if file.endswith('.xml'):
            path = os.path.join(images_dataset_path, file)
            xml_name = file.split('.')[0]

            tree = et.parse(path, parser)
            root = tree.getroot()

            gt_file.write(os.path.join(dataset_type, f'{xml_name}.jpg'))
            gt_file.write('\t')

            for persian_word in root.iter('name'):
                persian_word = persian_word.text.strip() if persian_word.text else ''
                latin_word = [l_word for l_word, p_word in words_dict.items() if p_word == persian_word]

                if len(latin_word):
                    gt_file.write(latin_word[0])
                else:
                    gt_file.write('')
            gt_file.write('\n')

    gt_file.close()


if __name__ == '__main__':
    prepare_dataset('', '')
