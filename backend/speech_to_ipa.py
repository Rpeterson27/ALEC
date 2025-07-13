from allosaurus.app import read_recognizer

# load your model
model = read_recognizer("interspeech21")

# run inference -> æ l u s ɔ ɹ s
print(model.recognize('allosaurus_test_french_mac_2.wav'))