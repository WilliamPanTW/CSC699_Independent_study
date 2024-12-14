# CSC699_Independent_study - Clapper Board Detection for Post-Production
This is a automated system that detects information from clapper Board(AKA Slate) in video and renames it according with user input. 

## DEMO
https://github.com/user-attachments/assets/5bb73674-3eaa-4002-bc16-2a8dbacc99fa


## Motivation
Renaming files manually is a tedious and time-consuming task, especially working on large cinema projects. This manual process consumes hours of valuable time and disrupts post-production workflows, leaving less time for creative storytelling and editing.

## What we want to solve 
 
- Reduce the time spent on manual file organization.
- Human make error 
- Lack of Automation Tools 

## Methodology
1. **YOLO model**: Detects the clapperboard in video frames, selects the frame with the highest confidence score, and extracts the bounding box.

2. **Image Cropping**: Crops the detected region of the clapperboard from the frame for better OCR.

3. **Text Extraction**: Uploads the cropped image to a Generative AI model to extract attributes (roll, scene, take) in JSON format.

4. **File Renaming**: Renames the video file using the extracted attributes for better organization.

## Reason to use LLM    
**Ease of Deployment** : Pre-trained LLMs provide ready-to-use solutions, saving development time.
**Cost-Effectiveness**: Using an existing LLM API can be cheaper than training a model from scratch.

## Challenges
- **Low Light Conditions**: Detecting and extracting text from poorly lit slates.  
- **Slate Variability**: Handling differences in slate designs, handwriting, and marker colors.  
- **Motion Blur**: Managing frames with blurry slate images due to movement.  
- **Complex Backgrounds**: Distinguishing slates from cluttered or visually similar backgrounds.  
- **OCR Limitations**: Processing unclear or poorly written text accurately.  
- **LLM Limitations**: Using an LLM API may pose potential data privacy risks and can be resource-intensive, consuming significant energy.

## Future Work
- Extend support for multilingual text detection.  
- Enhance robustness for low-quality footage and extreme lighting conditions.  
- Integrate the system with popular video editing software for seamless workflows.
- Use audio detection model to improve video frame synchronization 
- Build a local model to enhance privacy and energy efficiency, avoiding LLM reliance. 

## conclusion 
Although renaming files based on scene and take is not strictly required, but it is highly recommended because it helps with organization, efficiency, and communication during the post-production process.
