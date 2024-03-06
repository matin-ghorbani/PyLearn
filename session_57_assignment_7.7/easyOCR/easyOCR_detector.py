import easyocr as er


def predict_on_img(img_path: str, languages: list[str]) -> list:
    reader = er.Reader(languages)
    return reader.readtext(img_path)


def main() -> None:
    # Inference on a latin hand-writing text image
    # latin_path = './images/latin_hand_writing.jpg'
    # latin_result = predict_on_img(latin_path, ['en'])
    # print(f'{latin_result = }')

    # Inference on a persian hand-writing text image
    persian_path = './images/persian_hand_writing.jpg'
    persian_result = predict_on_img(persian_path, ['fa'])
    print(f'{persian_result = }')

    # Inference on a latin license plate image
    # latin_plate_path = './images/latin_plate.jpg'
    # latin_plate_result = predict_on_img(latin_plate_path, ['en'])
    # print(f'{latin_plate_result = }')

    # Inference on a persian license plate image
    # persian_plate_path = './images/persian_plate.jpg'
    # persian_plate_result = predict_on_img(persian_plate_path, ['fa'])
    # print(f'{persian_plate_result = }')


if __name__ == '__main__':
    main()
