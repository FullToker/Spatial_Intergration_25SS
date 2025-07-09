# Spatial Data Integration Projects

This repository contains three main spatial data integration and visualization projects for the 2025 Spring Semester, covering computer vision, data analysis, and web mapping applications.

## Projects Overview

### Task 1: Measure the cognitive load

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

### Task 2: Twitter VGI Analysis & Sentiment Comparison

**Directory**: `Task2/`

A comprehensive Twitter data analysis project focusing on Volunteered Geographic Information (VGI) research with advanced sentiment analysis capabilities. This project analyzes large-scale Twitter datasets from major sporting events and compares traditional ML approaches with modern Large Language Models.

#### Features
- **Large-scale Twitter Data Processing**: Handle 40M+ tweet JSON datasets with streaming parsers
- **Multi-event Analysis**: UEFA Champions League 2017 and Wimbledon 2017 datasets
- **Advanced Sentiment Analysis**: Compare BERT-based models vs OpenAI GPT-4 API
- **Geospatial Analysis**: Extract and process geographic coordinates and location data
- **Multi-language Support**: Process tweets in multiple languages with translation
- **Interactive Web Dashboard**: Real-time data visualization and exploration
- **Performance Benchmarking**: Compare speed, accuracy, and cost across methods

#### Key Components
- **Analysis Submodule** (`Analysis/`): Complete Twitter VGI analysis framework
  - `Load_Pre/`: Data preprocessing and loading modules
    - `read_Large_json.py`: Main Twitter JSON processor
    - `Sent_ana.ipynb`: BERT-based sentiment analysis
    - `geocsv_pre.ipynb`: Geocoding and CSV preprocessing
    - `data/`: Processed UEFA and Tennis datasets
  - `Eval_llm/`: LLM evaluation and comparison
    - `first_eval.ipynb`: ChatGPT vs traditional methods comparison
    - Analysis results and performance metrics
  - `Show_data/`: Web visualization interface
    - `tweet_analyzer.html`: Interactive analysis dashboard
    - `visual_json.ipynb`: Data visualization notebooks
  - `src/`: Generated plots and statistical analysis
- **Web Interface**: Direct dashboard access
  - `tweet_analyzer.html`: Main analysis dashboard
  - `main.js`, `script.js`, `worker.js`: Core application logic
  - `style.css`: UI styling and responsive design

#### Technologies Used
- **Python Stack**: pandas, numpy, transformers, torch, ijson
- **Machine Learning**: DistilBERT (SST-2), OpenAI GPT-4 API
- **Web Technologies**: HTML/CSS/JavaScript, Web Workers
- **Data Processing**: Streaming JSON parsing, geospatial analysis
- **Visualization**: Chart.js, matplotlib, interactive dashboards

#### Research Applications
- **Crisis Informatics**: Real-time event monitoring through social media
- **Digital Geography**: Understanding spatial patterns in social media discourse
- **Computational Social Science**: Automated content analysis method comparison
- **Sports Analytics**: Fan sentiment and engagement analysis during major events

---

### Task 3: Zoom Level Generalisation Conception

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

**Directory**: `Task3/my_web/`

A tile server and mapping application using MBTiles vector data with custom styling capabilities. This project provides a complete solution for serving vector map tiles with custom styling and visualization.

#### Features
- **Vector Tile Server**: MBTiles-based data serving with Switzerland map data
- **Custom Map Styling**: Comprehensive style configuration with Positron theme
- **Multi-format Support**: Vector tiles, sprites, and font glyph support
- **Web-based Preview**: HTML interface for style testing and visualization
- **Flexible Configuration**: JSON-based configuration for styles and data sources

#### Key Components
- `config.json`: Main server configuration for paths and data sources
- `data/`: Vector tile data storage
  - `switzerland.mbtiles`: Switzerland vector tile dataset
- `styles/`: Map styling configurations
  - `map.json`: Complete Positron-style map configuration with layers
- `sprites/`: Icon sprites for map symbols
  - `sprite.json`, `sprite.png`, `sprite@2x.png`: Icon definitions and assets
- `test_data/`: Testing environment with sample configurations
  - `config.json`: Test server configuration
  - `styles/`: Test style configurations (OSM Bright, MapTiler Basic)
  - `fonts/`: Font glyph collections (Open Sans variants)
  - `zurich_switzerland.mbtiles`: Zurich-focused test dataset

#### Development
1. Slice/Tile the geographical data
```bash
docker run -it --rm -v $(pwd):/data ghcr.io/systemed/tilemaker:master /data/monaco-latest.osm.pbf --output /data/monaco-latest.pmtiles
```
2. Run the Tiles server
```bash
docker run --rm -it -v /your/local/config/path:/data -p 8080:8080 maptiler/tileserver-gl:latest
```

#### Technologies Used
- **MBTiles**: Vector tile storage format
- **Mapbox GL JS**: Web mapping library for rendering
- **Vector Tiles**: Efficient geospatial data format
- **JSON Styling**: Mapbox Style Specification
- **PBF Fonts**: Protocol buffer font glyphs

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
├── Task2/                      # Twitter VGI Analysis & Sentiment Comparison
│   ├── Analysis/              # Twitter VGI analysis framework (submodule)
│   │   ├── Load_Pre/         # Data preprocessing and loading
│   │   ├── Eval_llm/         # LLM evaluation and comparison
│   │   ├── Show_data/        # Web visualization interface
│   │   └── src/              # Generated plots and statistics
│   ├── tweet_analyzer.html    # Main dashboard
│   ├── main.js               # Application logic
│   ├── script.js             # Data processing
│   ├── worker.js             # Background processing
│   └── style.css             # UI styling
├── Task3/                      # Interactive Web Mapping
│   ├── my_ol/                # OpenLayers application
│   │   ├── map_style/        # Custom map styles
│   │   ├── main.js           # Mapping logic
│   │   ├── style.css         # UI styling
│   │   └── index.html        # Main interface
│   └── my_web/               # MBTiles vector tile server
│       ├── config.json       # Server configuration
│       ├── data/             # Vector tile datasets
│       ├── styles/           # Map styling definitions
│       ├── sprites/          # Icon sprites and assets
│       └── test_data/        # Test configurations and datasets
└── README.md                   # This documentation
```

## Technical Integration

Each task demonstrates different aspects of spatial data integration:

1. **Task 1**: Computer vision and attention mechanisms for map understanding
2. **Task 2**: Advanced Twitter VGI analysis with ML/LLM sentiment comparison and geospatial processing
3. **Task 3**: Interactive web mapping and geospatial visualization
   - **my_ol/**: Client-side web mapping with OpenLayers
   - **my_web/**: Server-side vector tile serving with MBTiles

## Getting Started

Each task can be run independently:

- **Task 1**: Requires Python with PyTorch and computer vision libraries
- **Task 2**: Web-based application - open HTML files in browser
- **Task 3**: 
  - **my_ol/**: Node.js application with modern build tools
  - **my_web/**: MBTiles tile server (requires compatible tile server software)

## Contributing

This is an academic project repository for the 2025 Spring Semester. Each task represents progressive learning in spatial data integration techniques.

## License

Educational use only - 2025 Spring Semester Spatial Data Integration Course
