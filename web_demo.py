import gradio as gr
import os
import numpy as np
import torch
from util import save_image
import argparse
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision
from model.dualstylegan import DualStyleGAN
from model.encoder.psp import pSp
from PIL import Image
import cv2

def load_image(img):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    img = transform(img)
    return img.unsqueeze(dim=0)   



def get_place_number():
    dir_path = os.path.join(os.getcwd(),"data","buddha","images","train")
    place_number = {}
    for fpath in os.listdir(dir_path):
        fpath = fpath[:-4]
        fpath = fpath.split("-")
        place, number = fpath[0], fpath[1]
        if place in place_number.keys():
            place_number[place].append(number)
        else:
            place_number[place] = [number]
    return place_number

PLACE_NUMBER = get_place_number()

def generate_message(selection):
    buddha_num = len(PLACE_NUMBER[selection])
    return f"该地区共有{buddha_num}尊佛像，请输入范围在[0,{buddha_num-1}]间的数字"

def show_buddha_image(select_box, num_input_box):
    buddha_num = len(PLACE_NUMBER[select_box])
    if num_input_box >= buddha_num:
        raise gr.Error("请按提示输入数字！")
    image_fpath = str(select_box)+"-"+str(PLACE_NUMBER[select_box][int(num_input_box)])+".jpg"
    image_fpath = os.path.join(os.getcwd(),"data","buddha","images","train", image_fpath)
    image = Image.open(image_fpath)
    
    return image
    

## Stuff for style transfer
MODEL_PATH = "./checkpoint/"
MODEL_NAME = "generator-001500.pt"
STYLE = "buddha"
device = "cuda"
generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
generator.eval()
ckpt = torch.load(os.path.join(MODEL_PATH, STYLE, MODEL_NAME), map_location=lambda storage, loc: storage)
generator.load_state_dict(ckpt["g_ema"])
generator = generator.to(device)

model_path = os.path.join(MODEL_PATH, "encoder.pt")
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
if 'output_size' not in opts:
    opts['output_size'] = 1024    
opts = Namespace(**opts)
opts.device = device
encoder = pSp(opts)
encoder.eval()
encoder.to(device)

exstyles = np.load(os.path.join(MODEL_PATH, STYLE, "refined_exstyle_code.npy"), allow_pickle='TRUE').item()

z_plus_latent=False
return_z_plus_latent=False
input_is_latent=True    

print('Load models successfully!')



def process_image(style_image, content_image, select_box, num_input_box):
    if style_image is None or content_image is None:
        raise gr.Error("不存在佛像或人脸图片",duration=5)
    with torch.no_grad():
        I = load_image(content_image).to(device)
        img_rec, instyle = encoder(F.adaptive_avg_pool2d(I, 256), randomize_noise=False, return_latents=True, 
                                   z_plus_latent=z_plus_latent, return_z_plus_latent=return_z_plus_latent, resize=False)  
        img_rec = torch.clamp(img_rec.detach(), -1, 1)
        stylename = str(select_box)+"-"+str(PLACE_NUMBER[select_box][int(num_input_box)])+".jpg"
        latent = torch.tensor(exstyles[stylename]).to(device)
        
        exstyle = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)
        
        img_gen, _ = generator([instyle], exstyle, input_is_latent=input_is_latent, z_plus_latent=z_plus_latent,
                              truncation=0.75, truncation_latent=0, use_res=True, interp_weights=[1.0]*7+[1.0]*11)
        img_gen = torch.clamp(img_gen.detach(), -1, 1)
        np_img_gen = img_gen[0].cpu()
        tmp = ((np_img_gen.detach().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    tmp = cv2.resize(tmp, (300,300))
    return tmp
  




with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                select_box = gr.Radio(choices=list(PLACE_NUMBER.keys()), label="选择是哪个地区的佛", value="Tibet")
            with gr.Row():
                message_box = gr.Textbox(value="该地区共有94尊佛像，请输入范围在[0,93]间的数字",label="数字输入提示")
            with gr.Row():
                num_input_box = gr.Number(label="请按数字输入提示的信息输入数字")
          
                
        with gr.Column():
            show_buddha_button = gr.Button("展示佛像风格图片")
            style_image = gr.Image(label="佛像风格图片",interactive=False)
    with gr.Row():
        with gr.Column():
            content_image = gr.Image(label="人脸内容图片")
        with gr.Column():
            final_image = gr.Image(label="风格化图片")
   
        
 
    select_box.change(
        fn=generate_message, 
        inputs=[select_box], 
        outputs=message_box
    )
    show_buddha_button.click(
        fn=show_buddha_image,
        inputs=[select_box, num_input_box],
        outputs=style_image
    )
    
    
    
    run_button = gr.Button("运行")
    run_button.click(
        fn=process_image, 
        inputs=[style_image, content_image, select_box, num_input_box], 
        outputs=final_image
    )


demo.launch()
