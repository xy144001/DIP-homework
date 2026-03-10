import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """

    warped_image = np.array(image)
    ### FILL: Implement similarity MLS based image warping
    # Similarity MLS 至少需要 2 个控制点对
    if len(source_pts) < 2:
        return warped_image

    h, w_img = warped_image.shape[:2]

    # 使用逆向映射：对目标图像中的每个像素 v，求它在原图中的对应位置
    # 因此这里将 target 作为 q，source 作为 p
    p = np.array(source_pts, dtype=np.float32)
    q = np.array(target_pts, dtype=np.float32)

    # 1. 生成目标图像上所有像素的坐标网格 v
    grid_y, grid_x = np.mgrid[0:h, 0:w_img]
    v = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)

    # 2. 计算权重 w_i = 1 / (|v - q_i|^(2*alpha))
    v_expanded = v[:, np.newaxis, :]
    q_expanded = q[np.newaxis, :, :]
    dist_sq = np.sum((v_expanded - q_expanded) ** 2, axis=2)
    W = 1.0 / (dist_sq ** alpha + eps)

    # 3. 计算加权质心 q* 和 p*
    sum_W = np.sum(W, axis=1, keepdims=True)
    q_star = np.sum(W[:, :, np.newaxis] * q_expanded, axis=1) / sum_W
    p_star = np.sum(W[:, :, np.newaxis] * p[np.newaxis, :, :], axis=1) / sum_W

    # 4. 计算相对坐标
    q_hat = q_expanded - q_star[:, np.newaxis, :]
    p_hat = p[np.newaxis, :, :] - p_star[:, np.newaxis, :]
    v_hat = v - q_star

    # 5. 求每个像素处的 similarity 变换参数:
    #    M = [[a, b], [-b, a]]
    qx = q_hat[:, :, 0]
    qy = q_hat[:, :, 1]
    px = p_hat[:, :, 0]
    py = p_hat[:, :, 1]

    mu = np.sum(W * (qx * qx + qy * qy), axis=1, keepdims=True) + eps
    a = np.sum(W * (qx * px + qy * py), axis=1, keepdims=True) / mu
    b = np.sum(W * (qx * py - qy * px), axis=1, keepdims=True) / mu

    # 6. 映射回原图坐标: v_mapped = p* + (v - q*) @ M
    mapped_x = p_star[:, 0:1] + v_hat[:, 0:1] * a - v_hat[:, 1:2] * b
    mapped_y = p_star[:, 1:2] + v_hat[:, 0:1] * b + v_hat[:, 1:2] * a
    mapped_v = np.concatenate([mapped_x, mapped_y], axis=1)

    # 控制点处直接精确映射，避免数值误差
    exact_mask = dist_sq < eps
    if np.any(exact_mask):
        exact_rows, exact_cols = np.where(exact_mask)
        mapped_v[exact_rows] = p[exact_cols]

    # 7. 将坐标重组为图像网格尺寸
    map_x = mapped_v[:, 0].reshape((h, w_img)).astype(np.float32)
    map_y = mapped_v[:, 1].reshape((h, w_img)).astype(np.float32)

    # 8. 使用双线性插值进行重采样
    warped_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
