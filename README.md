# Spatial Data Integration Projects

This repository contains three main spatial data integration and visualization projects for the 2025 Spring Semester, covering computer vision, data analysis, and web mapping applications.

## Projects Overview

### Task 1: Vision Transformer Attention Visualization

**Directory**: `Task1/`

A comprehensive computer vision project implementing Vision Transformer (ViT) attention mechanisms for image analysis and visualization.

#### Features
- **ViT Attention Maps**: Visualize attention patterns in Vision Transformers
- **Multi-layer Analysis**: Examine attention across different transformer layers
- **Segmentation Integration**: Combine attention with image segmentation
- **Max Fusion**: Advanced attention fusion techniques
- **BubbleView Integration**: Interactive attention visualization using BubbleView framework

#### Key Components
- `vit_attention_maps.py`: Core attention visualization implementation
- `vit_maxFusion.py`: Maximum fusion attention techniques
- `vit_seg_attention.py`: Segmentation-based attention analysis
- `vit_vis.py`: General visualization utilities
- `bubbleview-master/`: Interactive attention visualization framework
- `src/`: Generated attention maps and analysis results

#### Technologies Used
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformer models
- **OpenCV**: Image processing
- **Matplotlib**: Visualization
- **BubbleView**: Interactive attention visualization

---

### Task 2: Tweet Data Analysis Dashboard

**Directory**: `Task2/`

A web-based data analysis application for processing and visualizing tweet data with statistical analysis capabilities.

#### Features
- **Tweet Data Processing**: Parse and analyze tweet datasets
- **Statistical Analysis**: Generate comprehensive statistics and metrics
- **Interactive Dashboard**: Real-time data visualization
- **Multi-threading**: Web worker implementation for performance
- **Responsive Design**: Mobile-friendly interface

#### Key Components
- `tweet_analyzer.html`: Main analysis dashboard
- `main.js`: Core application logic
- `script.js`: Data processing functions
- `worker.js`: Web worker for background processing
- `style.css`: UI styling and responsive design
- `new.html`: Additional interface components

#### Technologies Used
- **HTML/CSS/JavaScript**: Frontend implementation
- **Web Workers**: Background data processing
- **Chart.js**: Data visualization
- **Responsive Design**: Cross-platform compatibility

---

### Task 3: OpenLayers Interactive Map Application

**Directory**: `Task3/my_ol/`

An advanced web mapping application built with OpenLayers and MapLibre GL JS, featuring custom styling and interactive geospatial features.

#### Features
- **Custom Map Styling**: Geoapify API integration with custom themes
- **Interactive Features**: Click-to-query functionality with property display
- **Real-time Information**: Dynamic coordinate and zoom level display
- **Navigation Controls**: Full map interaction capabilities
- **Responsive Design**: Cross-device compatibility

#### Key Components
- `index.html`: Main application interface
- `main.js`: Core mapping logic and interactions
- `style.css`: UI styling and layout
- `map_style/`: Custom map styling configurations
  - `final.json`: Production map style
  - `map.json`: Base map configuration
- `package.json`: Dependencies and build configuration
- `vite.config.js`: Build tool configuration

#### Technologies Used
- **MapLibre GL JS**: Core mapping library
- **Geoapify API**: Map tiles and styling services
- **Vite**: Modern build tool
- **GitHub Actions**: CI/CD pipeline
- **Dependabot**: Automated dependency management

#### Development Setup
```bash
cd Task3/my_ol/
npm install
npm run dev
```

---

## Repository Structure
```
SDI/
├── Task1/                      # Vision Transformer Attention Analysis
│   ├── vit_attention_maps.py   # Core attention visualization
│   ├── vit_maxFusion.py       # Maximum fusion techniques  
│   ├── vit_seg_attention.py   # Segmentation attention
│   ├── vit_vis.py            # Visualization utilities
│   ├── bubbleview-master/     # Interactive visualization framework
│   └── src/                   # Generated analysis results
├── Task2/                      # Tweet Data Analysis
│   ├── tweet_analyzer.html    # Main dashboard
│   ├── main.js               # Application logic
│   ├── script.js             # Data processing
│   ├── worker.js             # Background processing
│   └── style.css             # UI styling
├── Task3/                      # Interactive Web Mapping
│   └── my_ol/                # OpenLayers application
│       ├── map_style/        # Custom map styles
│       ├── main.js           # Mapping logic
│       ├── style.css         # UI styling
│       └── index.html        # Main interface
└── README.md                   # This documentation
```

## Technical Integration

Each task demonstrates different aspects of spatial data integration:

1. **Task 1**: Computer vision and attention mechanisms for spatial understanding
2. **Task 2**: Data analysis and statistical processing of location-based social media data
3. **Task 3**: Interactive web mapping and geospatial visualization

## Getting Started

Each task can be run independently:

- **Task 1**: Requires Python with PyTorch and computer vision libraries
- **Task 2**: Web-based application - open HTML files in browser
- **Task 3**: Node.js application with modern build tools

## Contributing

This is an academic project repository for the 2025 Spring Semester. Each task represents progressive learning in spatial data integration techniques.

## License

Educational use only - 2025 Spring Semester Spatial Data Integration Course
