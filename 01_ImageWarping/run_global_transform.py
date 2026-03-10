import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    transformed_image = np.array(image)

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    ### FILL: Apply Composition Transform 
    # 获取填充后图像的尺寸和中心点
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # 1. 旋转与缩放矩阵 (围绕图像中心)
    # cv2.getRotationMatrix2D 直接生成围绕指定中心的旋转和缩放 2x3 矩阵
    M_rot_scale_2x3 = cv2.getRotationMatrix2D((cx, cy), rotation, scale)
    # 使用作业提供的函数将其转换为 3x3 齐次矩阵
    M_rot_scale = to_3x3(M_rot_scale_2x3)

    # 2. 水平翻转矩阵 (围绕图像中心)
    # 如果发生翻转，X坐标的变化为: x_new = w - x (即围绕中心线对称)
    if flip_horizontal:
        M_flip = np.array([
            [-1, 0, w],
            [ 0, 1, 0],
            [ 0, 0, 1]
        ], dtype=np.float32)
    else:
        M_flip = np.eye(3, dtype=np.float32) # 不翻转则为单位矩阵

    # 3. 平移矩阵
    M_trans = np.array([
        [1, 0, translation_x],
        [0, 1, translation_y],
        [0, 0, 1]
    ], dtype=np.float32)

    # 4. 矩阵复合 (Composition)
    # 按照 旋转缩放 -> 翻转 -> 平移 的顺序进行矩阵乘法 (@ 是 numpy 中的矩阵乘法运算符)
    # 注意：矩阵乘法作用于列向量时，是从右向左计算的
    M_composed = M_trans @ M_flip @ M_rot_scale

    # 5. 提取最终的 2x3 仿射矩阵并应用
    M_final = M_composed[:2, :]
    # 使用 cv2.warpAffine 进行图像变换，borderValue 设置边缘用白色填充
    transformed_image = cv2.warpAffine(image, M_final, (w, h), borderValue=(255, 255, 255))

    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
