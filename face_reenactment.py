# coding: utf-8
# clone repo https://github.com/cleardusk/3DDFA_V2
# add this file in it 
# run python face_reenactment.py -s /path/to/source.jpg -t path/to/target.jpg
__author__ = 'cleardusk'

import sys
import argparse
import cv2
import yaml
import numpy as np
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.render import render  

def extract_facial_parameters(img_fp, tddfa, face_boxes):
    """Extract 3DMM parameters (facial expressions) from the source image."""
    img = cv2.imread(img_fp)
    boxes = face_boxes(img)
    if len(boxes) == 0:
        print(f'No face detected in the source image, exit')
        sys.exit(-1)
    param_lst, _ = tddfa(img, boxes)
    return param_lst[0]  # Assuming only one face in the image


def reconstruct_3d_face(img_fp, tddfa, face_boxes):
    """Reconstruct the 3D face model from the target image."""
    img = cv2.imread(img_fp)
    boxes = face_boxes(img)
    if len(boxes) == 0:
        print(f'No face detected in the target image, exit')
        sys.exit(-1)
    param_lst, roi_box_lst = tddfa(img, boxes)
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
    return ver_lst[0], param_lst[0]  # Assuming only one face in the image


def transfer_expressions(source_params, target_ver, target_params, tddfa):
    """Transfer facial expressions from the source to the target face."""
    # Transfer the expression parameters from the source to the target
    target_params[56:] = source_params[56:]  

    # Calculate the bounding box from the target_ver (3D vertices)
    x_min, y_min = np.min(target_ver, axis=1)[:2]
    x_max, y_max = np.max(target_ver, axis=1)[:2]
    roi_box = [x_min, y_min, x_max, y_max]  # Approximate roi_box from the 3D vertices

    # Reconstruct the 3D face with the new parameters
    new_ver = tddfa.recon_vers([target_params], [roi_box], dense_flag=True)
    return new_ver[0]

def render_result(img, ver, tddfa, wfp):
    """Render the target face with the transferred expressions."""
    render(img, [ver], tddfa.tri, alpha=0.6, show_flag=True, wfp=wfp)


def visualize_masks(source_ver, target_ver, transferred_ver, tddfa):
    """
    Visualize the 3D masks of the source, target, and transferred faces.
    """
    # Create a blank image for rendering the masks
    img_size = (500, 500, 3)  # Size of the output image
    blank_img = np.zeros(img_size, dtype=np.uint8)

    # Render the source mask
    source_mask = render(blank_img.copy(), [source_ver], tddfa.tri, alpha=1.0, show_flag=False)
    source_mask = cv2.cvtColor(source_mask, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

    # Render the target mask
    target_mask = render(blank_img.copy(), [target_ver], tddfa.tri, alpha=1.0, show_flag=False)
    target_mask = cv2.cvtColor(target_mask, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

    # Render the transferred mask
    transferred_mask = render(blank_img.copy(), [transferred_ver], tddfa.tri, alpha=1.0, show_flag=False)
    transferred_mask = cv2.cvtColor(transferred_mask, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

    # Create a single figure with three subplots
    plt.figure(figsize=(15, 5))

    # Plot the source mask
    plt.subplot(1, 3, 1)
    plt.imshow(source_mask)
    plt.title("Source Mask")
    plt.axis('off')

    # Plot the target mask
    plt.subplot(1, 3, 2)
    plt.imshow(target_mask)
    plt.title("Target Mask")
    plt.axis('off')

    # Plot the transferred mask
    plt.subplot(1, 3, 3)
    plt.imshow(transferred_mask)
    plt.title("Transferred Mask")
    plt.axis('off')

    # Show the figure
    plt.tight_layout()
    plt.show()

def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Extract facial parameters from the source image
    source_img = cv2.imread(args.source_img_fp)
    source_boxes = face_boxes(source_img)
    if len(source_boxes) == 0:
        print(f'No face detected in the source image, exit')
        sys.exit(-1)
    source_params, source_roi_box = tddfa(source_img, source_boxes)
    source_ver = tddfa.recon_vers(source_params, source_roi_box, dense_flag=True)[0]

    # Reconstruct 3D face from the target image
    target_img = cv2.imread(args.target_img_fp)
    target_boxes = face_boxes(target_img)
    if len(target_boxes) == 0:
        print(f'No face detected in the target image, exit')
        sys.exit(-1)
    target_params, target_roi_box = tddfa(target_img, target_boxes)
    target_ver = tddfa.recon_vers(target_params, target_roi_box, dense_flag=True)[0]

    # Transfer expressions from source to target
    transferred_params = target_params[0].copy()  # Copy target parameters
    transferred_params[56:] = source_params[0][56:]  # Transfer expression parameters
    transferred_ver = tddfa.recon_vers([transferred_params], [target_roi_box[0]], dense_flag=True)[0]

    # Visualize the masks
    visualize_masks(source_ver, target_ver, transferred_ver, tddfa)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of face reenactment using 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-s', '--source_img_fp', type=str, default='examples/inputs/source.jpg', help='Path to the source image')
    parser.add_argument('-t', '--target_img_fp', type=str, default='examples/inputs/target.jpg', help='Path to the target image')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
