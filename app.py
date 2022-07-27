# Portrait Photo Generator App

# Imports
from PIL import Image, ImageFilter
import numpy as np
from transformers import pipeline
import gradio as gr
import os

model = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")

pred = []

def img_resize(image):
    width = 1280
    width_percent = (width / float(image.size[0]))
    height = int((float(image.size[1]) * float(width_percent)))
    return image.resize((width, height))

def image_objects(image):
    global pred
    image = img_resize(image)
    pred = model(image)
    pred_object_list = [str(i)+'_'+x['label'] for i, x in enumerate(pred)]
    return gr.Dropdown.update(choices = pred_object_list, interactive = True)

def blurr_object(image, object, blur_strength):
    image = img_resize(image)

    object_number = int(object.split('_')[0])
    mask_array = np.asarray(pred[object_number]['mask'])/255
    image_array = np.asarray(image)

    mask_array_three_channel = np.zeros_like(image_array)
    mask_array_three_channel[:,:,0] = mask_array
    mask_array_three_channel[:,:,1] = mask_array
    mask_array_three_channel[:,:,2] = mask_array

    segmented_image = image_array*mask_array_three_channel

    blur_image = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=blur_strength)))
    mask_array_three_channel_invert = 1-mask_array_three_channel
    blur_image_reverse_mask = blur_image*mask_array_three_channel_invert

    blurred_output_image = Image.fromarray((blur_image_reverse_mask).astype(np.uint8)+segmented_image.astype(np.uint8))

    return blurred_output_image

app = gr.Blocks()


with app:
    gr.Markdown(
            """
            ## Portrait Photo Generator
            - Create stunning portrait photos by blurring the background of your selected object.
            - Adjust the blurring strength using the slider.
            """)
    with gr.Row():
        with gr.Column():
            gr.Markdown(
            """
            ### Input Image
            """)
            image_input = gr.Image(type="pil")
            

        with gr.Column():
            with gr.Row():
                gr.Markdown(
                """
                ### Found Objects
                """)
            with gr.Row():
                blur_slider = gr.Slider(minimum=0.5, maximum=10, value=3, label="Adject Blur Strength")
            with gr.Row():
                object_output = gr.Dropdown(label="Select Object From Dropdown")

    
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ### Blurred Image Output
                """)
            image_output = gr.Image()
        with gr.Column():
            gr.Markdown(
                """
                ### Example Images
                """)
            gr.Examples(
                examples=[
                "test_images/dog_horse_cowboy.jpg",
                "test_images/woman_and_dog.jpg",
                "test_images/family_in_sofa.jpg",
                "test_images/people_group.jpg"
                ],
                fn=image_objects,
                inputs=image_input, 
                outputs=object_output)

    image_input.change(fn=image_objects, 
        inputs=image_input, 
        outputs=object_output
        )

    object_output.change(fn=blurr_object, 
        inputs=[image_input, object_output, blur_slider],
        outputs=image_output)

    blur_slider.change(fn=blurr_object, 
        inputs=[image_input, object_output, blur_slider],
        outputs=image_output)
    

app.launch()