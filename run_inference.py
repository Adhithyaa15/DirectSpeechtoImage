import torch
from load_pipeline import load

pipe = load("combined_speech2image_checkpoint.pt")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

image = pipe("audio.wav").images[0]
image.save("output.png")
