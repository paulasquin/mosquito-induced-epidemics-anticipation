# testing command_classification.py
import image_recognition.inception_classification.command_classification as command_classification
import sys

PATH_TEST_IMG = "pic_014_cropped.jpg"


def test_command_retrain():
    command_classification.train_and_monitor()


def test_label_image(path_img):
    print(command_classification.label_automatic(path_img))


if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "--retrain":
        print("Testing command_retrain")
        test_command_retrain()
    elif len(sys.argv) > 1 and sys.argv[1] == "--label":
        print("Testing image labelling")
        if len(sys.argv) > 2:
            for path_img in sys.argv[2:]:
                test_label_image(path_img)
        else:
            test_label_image(PATH_TEST_IMG)
    else:
        if len(sys.argv) > 1 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
            print("Didn't recognize command : " + sys.argv[1])
        print("Usage :")
        print("--retrain : retrain the inception model")
        print("--label [optional path to one or more images]")
