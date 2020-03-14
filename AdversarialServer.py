import os, json, io, traceback
import base64

import torch
import torchvision
from PIL import Image

page = '''
<form action="/" method="post" enctype="multipart/form-data">
  <input type="file" name="unmodified.png" accept="image/png"></br>
  <input type="submit" value="Submit"><!--v=vSt4Az6oXO0--></br>
  
  Submit an image to generate an adversarial version! Targets <a href="https://arxiv.org/abs/1611.05431">ResNeXt50</a>.</br>
  Your image probably won't get classified well to begin with if it isn't a photo of resolution >299x299 and of something in the ImageNet dataset.</br>
  The image below is the last one somebody submitted.</br></br>
  
  {6}
  
  Before:</br>
  <img width="299" height="299" alt="{0}" src="data:image/png;base64,{1}" /></br>
  {2:10.2f}% {0}</br>
  </br>
  After:</br>
  <img width="299" height="299" alt="{3}" src="data:image/png;base64,{4}" /></br>
  {5:10.2f}% {3}</br>
</form>
    '''
with open("ImageNetClasses.json", "r") as jfile:
  labels = json.load(jfile)

network = torchvision.models.resnext50_32x4d(pretrained=True)
network.eval()

mean = torch.as_tensor([0.485, 0.456, 0.406])
std_dev = torch.as_tensor([0.229, 0.224, 0.225])

def application(request, start_response):
    error_message = ""

    try:
        path = request.get("PATH_INFO")
    except ValueError:
        path = "/"
    if path != "/":
        start_response('404 Not Found', [('Content-Type','text/html')])
        return['Nothing to see here, folks.']

    try:
        length = int(request.get('CONTENT_LENGTH', '0'))
    except ValueError:
        length = 0

    try:

        if length != 0:
            posted = request['wsgi.input'].read()
            separator_end = posted.index(b'\r\n')
            separator = posted[:separator_end]

            image_start = posted.index(b'\r\n\r\n') + 4
            image = posted[image_start:]
            image_end = image.index(separator) - 2
            image = image[:image_end]
            with io.BytesIO(image) as imfile:
                uploaded = Image.open(imfile)
                if uploaded.mode not in ["RGB", "RGBA"]:
                    raise ValueError("Invalid image format - RGB or RGBA PNG only")
                if uploaded.mode == "RGBA":
                    uploaded = uploaded.convert("RGB")
                width, height = uploaded.size
                ratio = max(299.0 / width, 299.0 / height)
                new_width, new_height = ratio * width, ratio * height
                uploaded = uploaded.resize((int(new_width), int(new_height)))
                crop = True
                if width > height:
                    left = (new_width - 299) // 2
                    right = (new_width + 299) // 2
                    top = 0
                    bottom = 299
                elif height > width:
                    left = 0
                    right = 299
                    top = (new_height - 299) // 2
                    bottom = (new_height + 299) // 2
                else:
                    crop = False
                if crop:
                    uploaded = uploaded.crop((left, top, right, bottom))

                uploaded.save("unmodified.png")

                tensor = torchvision.transforms.ToTensor()(uploaded)
                tensor = torchvision.transforms.Normalize(mean=mean,
                                                      std=std_dev)(tensor)
                tensors = [tensor.unsqueeze(0)]
                tensors[0].requires_grad = True
                scores = network(tensors[0])
                base_prediction = scores.max(1, keepdim=True)[1][0] #torch.tensor([882])#scores.max(1, keepdim=True)[1][0]
                for i in range(4):
                    # Assume the output class prediction was correct
                    loss = torch.nn.functional.nll_loss(scores, base_prediction)
                    network.zero_grad()
                    loss.backward()
                    image_gradients = tensors[-1].grad.data
                    gradient_signs = image_gradients.sign()
                    with torch.no_grad():
                        tensors.append(tensors[-1] + 3.0 * gradient_signs / 255.0)
                    tensors[-1].requires_grad = True
                    scores = network(tensors[-1])
               
                # Denormalize
                tensor = tensors[-1]
                tensor.requires_grad = False
                tensor.mul_(std_dev[:, None, None]).add_(mean[:, None, None])
                tensor = torch.clamp(tensor[0], 0, 1)
                adv_im = torchvision.transforms.ToPILImage(mode='RGB')(tensor)
                adv_im.save('adversarial.png')
    except Exception as e:
        error_message = '<p style="color:red"><b>Something went wrong! Please try again. Error below:</br>' + str(e) + "</br>" + traceback.format_exc().replace("\n", "</br>") + '</b></p>'

    with open("./unmodified.png", "br") as base_file:
        img = Image.open("./unmodified.png")
        if img.mode == "RGBA":
            img = img.convert("RGB")

        tensor = torchvision.transforms.ToTensor()(img)
        tensor = torchvision.transforms.Normalize(mean=mean,
                                                  std=std_dev)(tensor)
        tensor = tensor.unsqueeze(0)
        scores = torch.nn.Softmax(dim=1)(network(tensor))
        base_top_score, base_top_class = torch.max(scores, 1)

        base_image = base64.b64encode(base_file.read()).decode()
    with open("./adversarial.png", "br") as adv_file:
        img = Image.open("./adversarial.png")
        if img.mode == "RGBA":
            img = img.convert("RGB")
        tensor = torchvision.transforms.ToTensor()(img)
        tensor = torchvision.transforms.Normalize(mean=mean,
                                                  std=std_dev)(tensor)
        tensor = tensor.unsqueeze(0)
        scores = torch.nn.Softmax(dim=1)(network(tensor))
        adv_top_score, adv_top_class = torch.max(scores, 1)

        adv_image = base64.b64encode(adv_file.read()).decode()

    start_response('200 OK', [('Content-Type','text/html')])
    return [page.format(labels[int(base_top_class.item())],
                        base_image,
                        100.0 * base_top_score.item(),
                        labels[int(adv_top_class.item())],
                        adv_image,
                        100.0 * adv_top_score.item(),
                        error_message).encode()]
