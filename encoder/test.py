from inference import EncoderInference


encoder = EncoderInference()
encoder.load_model("encoder.pt")

tmp = encoder.predict("../data/84-121123-0000.flac")
print(tmp.shape)
