# AI-image-caption-pipeline                                                                                
                                                                                                           
This repository contains a multimodal AI image caption generator based on YOLO and Graph Neural Network    
(GNN), with future plans to incorporate Vision Transformers (ViT). The project demonstrates how to generate
descriptive captions for images by combining object detection and graph-based reasoning.                   
                                                                                                           
## Features                                                                                                
                                                                                                           
- **Object Detection**: Uses YOLO to detect objects in images.                                             
- **Graph Neural Network**: Models relationships between detected objects to improve caption quality.      
- **Pipeline Orchestration**: Modular design for easy extension and experimentation.                       
- **Future Scope**: Plans to integrate Vision Transformers (ViT) for enhanced performance.                 
                                                                                                           
## Repository Structure                                                                                    
                                                                                                           
```                                                                                                        
AI-image-caption-pipeline/                                                                                 
├── builder.py             # Pipeline builder and main orchestration logic                                 
├── g-neural-network.py    # Graph Neural Network implementation                                           
├── runner                 # Entry point script to run the pipeline                                        
```                                                                                                        
                                                                                                           
## Getting Started                                                                                         
                                                                                                           
### Prerequisites                                                                                          
                                                                                                           
- Python 3.8+                                                                                              
- pip                                                                                                      
                                                                                                           
### Installation                                                                                           
                                                                                                           
1. Clone the repository:                                                                                   
   ```bash                                                                                                 
   git clone https://github.com/Aryo15/AI-image-caption-pipeline.git                                       
   cd AI-image-caption-pipeline                                                                            
   ```                                                                                                     
                                                                                                           
2. Install dependencies:                                                                                   
   ```bash                                                                                                 
   pip install -r requirements.txt                                                                         
   ```                                                                                                     
                                                                                                           
### Usage                                                                                                  
                                                                                                           
To run the full image captioning pipeline, use:                                                            
```bash                                                                                                    
python runner                                                                                              
```                                                                                                        
This will process the input image(s), perform object detection, build the object relationship graph, and   
generate captions.                                                                                         
                                                                                                           
You can also run individual modules for development or testing:                                            
- `builder.py`: Main pipeline logic and integration.                                                       
- `g-neural-network.py`: GNN model and utilities.                                                          
                                                                                                           
## Contributing                                                                                            
                                                                                                           
Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.    
                                                                                                           
## License                                                                                                 
                                                                                                           
This project is licensed under the MIT License.  
