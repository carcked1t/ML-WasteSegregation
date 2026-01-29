# webcam_predict.py
import cv2
from PIL import Image
import torch
import torchvision.transforms as T

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load("waste_model_scripted.pt")
model.to(device)
model.eval()

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Transform
transform = T.Compose([
    T.Resize((160, 160)),
    T.ToTensor()
])

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV) to RGB (PIL)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = f"{CLASSES[pred.item()]}: {conf.item():.2f}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam Waste Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
