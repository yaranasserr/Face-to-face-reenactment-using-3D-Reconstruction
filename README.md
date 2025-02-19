# Face-to-face-reenactment-using-3D-Reconstruction


![Results](https://github.com/yaranasserr/Face-to-face-reenactment-using-3D-Reconstruction/blob/main/3D.png)

This repository utilizes code from [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2), which employs 3D Morphable Models (3DMM) for 3D face reconstruction.

## Overview
This project adapts the 3D reconstruction code from 3DDFA to perform **face-to-face reenactment**.  
- The **source image** is used for expression encoding.  
- The **target image** is used for 3D face reconstruction.  

## Face Landmarks (Geometric Structure)
Face landmarks define the geometric structure of the face, including key points for:  
- **Eyes**  
- **Eyebrows**  
- **Nose**  
- **Mouth**  
- **Jawline**  
- **Chin**  

These landmarks provide the foundational shape of the face in 3D space, helping to model the structure and position of facial features.

## Facial Expressions
Facial expressions represent dynamic changes in the face due to muscle movements (e.g., smiling, frowning, raising eyebrows). These expressions are encoded as parameters in **3D Morphable Models (3DMM)** and modify the neutral face shape.

---

## Step-by-Step Adaptation
### 1. Extract the 3DMM Parameters from the Source Image
- Load the **source image** and detect the face.
- Use **TDDFA** to extract the **3DMM parameters** (pose, shape, expression) of the source face.

### 2. Extract Landmarks from the Target Image
- Detect the **target image** face.
- Use **TDDFA** to obtain landmarks and the bounding box for the target face.

### 3. Transfer the Expression Parameters to the Target Image
- Apply the expression parameters from the **source image** to the **target image**, while keeping the **shape and pose parameters** from the target’s landmarks.

### 4. Render the 3D Mask on the Target Image
- After applying the **facial expression** from the source, reconstruct the **3D vertices** and render the **3D mask** on the target image.

---

## How to Use the Code
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/cleardusk/3DDFA_V2.git
2 . **Add the face reenactment script**
- Place the face_reenactment.py script inside the 3DDFA_V2 directory.
3.**Add the Source and Target Images**
- Place your source and target images in a specified folder.
- Run the script
 ```bash
python /path/to/3DDFA_V2/face_reenactment.py \
--source_img_fp /path/to/source_image.jpg \
--target_img_fp /path/to/target_image.jpg \
--output_fp /path/to/output_result.jpg \
--mode <mode> \
--show_flag <true/false>
```

